"""MemoryUpdater：把通过 Judge 的候选记忆路由到 3 个 Memory Service。
不反向引用 container，依赖通过构造函数显式注入（避免循环导入）。

路由规则（和新的 3 类 MemoryType 严格对齐）：
 PROFILE      → ProfileService.upsert_patch(user_id, metadata.patch 或 dict(content))
 RELATIONSHIP → RelationshipService.add_or_update(user_id, canonical_name, aliases, relation, extra, confidence, person_id)
 EPISODIC     → EpisodicService.add(user_id, content, tags, person_ids, metadata, timestamp, importance)
 其他类型     → 打印WARN + 跳过（例如已删除的 semantic/emotion）
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from loguru import logger

from services.memory.memory_candidate import MemoryCandidate, MemoryType
from services.memory.episodic_service import EpisodicService
from services.memory.relationship_service import RelationshipService
from services.memory.profile_service import ProfileService


class MemoryUpdater:
    def __init__(
        self,
        profile_service: ProfileService,
        relationship_service: RelationshipService,
        episodic_service: EpisodicService,
    ) -> None:
        self.profile_svc = profile_service
        self.relation_svc = relationship_service
        self.episodic_svc = episodic_service

    # ============== public ==============
    def update(self, user_id: str, memories: List[MemoryCandidate]) -> None:
        if not memories:
            return
        for m in memories:
            try:
                self._route_one(user_id, m)
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    f"[Updater] 写入失败: type={m.memory_type} content={m.content!r} err={e}"
                )

    # ============== routing ==============
    def _route_one(self, user_id: str, m: MemoryCandidate) -> None:
        t = m.memory_type
        if t is MemoryType.PROFILE:
            self._write_profile(user_id, m)
        elif t is MemoryType.RELATIONSHIP:
            self._write_relationship(user_id, m)
        elif t is MemoryType.EPISODIC:
            self._write_episodic(user_id, m)
        else:
            # 旧的 semantic/emotion 或未知类型，显式跳过。
            logger.warning(
                f"[Updater] 跳过未知/已删除的记忆类型: {t!r} content={m.content!r}"
            )

    # ---------- PROFILE ----------
    def _write_profile(self, user_id: str, m: MemoryCandidate) -> None:
        patch: Dict[str, Any] = m.meta("patch") or {}
        replace_full: bool = bool(m.meta("replace", False))
        if replace_full:
            self.profile_svc.replace(user_id, patch or {"raw": m.content})
            logger.info(f"[Updater] PROFILE.replace user_id={user_id}")
            return
        # 没有 patch 时，做一个兜底：用自然语言 content 塞到 metadata.raw_hint 里，
        # 避免写空 dict。等后续 LLM Extractor 完善就会一直给 patch，这里仅兼容过渡。
        if not patch:
            patch = {"__raw_hint": m.content}
        merged = self.profile_svc.upsert_patch(user_id, patch)
        logger.info(f"[Updater] PROFILE.upsert user_id={user_id} keys={list(patch.keys())}")
        _ = merged

    # ---------- RELATIONSHIP ----------
    def _write_relationship(self, user_id: str, m: MemoryCandidate) -> None:
        canonical_name: str = m.meta("canonical_name") or m.content.strip()
        aliases: List[str] = list(m.meta("aliases") or [])
        # 如果 content 本身不是 canonical_name（例如 content 是别名），也加入 aliases
        alt = m.content.strip()
        if alt and alt.lower() != canonical_name.lower() and alt not in aliases:
            aliases.append(alt)
        relation: str = str(m.meta("relation") or "unknown")
        extra: Dict[str, Any] = dict(m.meta("extra") or {})
        confidence: float = float(m.meta("confidence") or max(0.5, min(1.0, float(m.importance or 0.5))))
        person_id: Optional[str] = m.meta("person_id") or None
        row = self.relation_svc.add_or_update(
            user_id=user_id,
            canonical_name=canonical_name,
            aliases=aliases,
            relation=relation,
            extra=extra,
            confidence=confidence,
            person_id=person_id,
        )
        logger.info(
            f"[Updater] RELATIONSHIP person_id={row.get('person_id')} "
            f"name={row.get('canonical_name')!r} relation={row.get('relation')}"
        )

    # ---------- EPISODIC ----------
    def _write_episodic(self, user_id: str, m: MemoryCandidate) -> None:
        # =========================================================
        # 1. 时间
        # =========================================================

        ts_val = m.meta("timestamp")

        ts: Optional[_dt.datetime]

        if isinstance(ts_val, _dt.datetime):
            ts = ts_val

        elif isinstance(ts_val, str) and ts_val:
            try:
                ts = _dt.datetime.fromisoformat(
                    ts_val.replace("Z", "+00:00")
                )
            except Exception:  # noqa: BLE001
                ts = None

        else:
            ts = None

        # =========================================================
        # 2. 人物关联
        # =========================================================

        # Extractor 已经明确知道的 person_id
        person_ids: List[str] = list(
            m.meta("person_ids") or []
        )

        # Extractor 提取出来的人物名称
        person_names: List[str] = list(
            m.meta("person_names") or []
        )

        # 根据人物名称解析 person_id
        for person_name in person_names:

            if not person_name:
                continue

            try:
                person = self.relation_svc.resolve_name(
                    user_id=user_id,
                    name=person_name,
                )

                if not person:
                    logger.debug(
                        f"[Updater] 未找到人物: "
                        f"user_id={user_id}, name={person_name!r}"
                    )
                    continue

                person_id = person.get("person_id")

                if person_id and person_id not in person_ids:
                    person_ids.append(person_id)

                    logger.debug(
                        f"[Updater] EPISODIC 人物关联: "
                        f"{person_name!r} -> {person_id}"
                    )

            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"[Updater] 人物解析失败: "
                    f"name={person_name!r}, err={e}"
                )

        # =========================================================
        # 3. metadata
        # =========================================================

        metadata: Dict[str, Any] = dict(
            m.metadata or {}
        )

        # 如果 Extractor 没显式写 metadata.emotion，
        # 但其他流程已经提供 emotion，则兼容写入
        if (
                "emotion" not in metadata
                and m.meta("emotion")
        ):
            metadata["emotion"] = m.meta("emotion")

        # =========================================================
        # 4. importance
        # =========================================================

        importance = float(
            m.importance or 0.5
        )

        # =========================================================
        # 5. 写入 Episodic
        # =========================================================

        eid = self.episodic_svc.add(
            user_id=user_id,
            content=m.content,
            tags=m.tags or [],
            person_ids=person_ids,
            metadata=metadata,
            timestamp=ts,
            importance=importance,
        )

        logger.info(
            f"[Updater] EPISODIC "
            f"id={eid} "
            f"persons={person_ids} "
            f"tags={m.tags or []} "
            f"importance={importance:.2f}"
        )
