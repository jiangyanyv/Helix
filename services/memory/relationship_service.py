"""RelationshipService：人物实体索引 + 关系 + 别名匹配 + 实体合并。

不 preload 全量；按 user_id 为粒度 LRU 缓存正常(status=1)人物列表；别名单独 TTL 缓存。
"""
#
from __future__ import annotations

import datetime as _dt
import json
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cachetools import LRUCache, TTLCache
from loguru import logger
from sqlalchemy import text

from config import Config
from infrastructure.database.mysql import get_db_session

PERSON_STATUS_ACTIVE = 1
PERSON_STATUS_MERGED = 2
PERSON_STATUS_DISCARDED = 3


def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:  # noqa: BLE001
            return value
    return value


def _row_to_dict(row: Any) -> Dict[str, Any]:
    d = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
    for col in ("aliases_json", "extra_json"):
        if col in d:
            d[col] = _parse_json(d[col])
    if "confidence" in d and d["confidence"] is not None:
        d["confidence"] = float(d["confidence"])
    return d


class RelationshipService:
    def __init__(self) -> None:
        # L1：user_id -> {person_id: row_dict} 仅 status=1 正常人物
        self._persons_lru: LRUCache[str, Dict[str, Dict[str, Any]]] = LRUCache(
            maxsize=Config.RELATIONSHIP_LRU_MAXSIZE
        )
        # L2：(user_id, alias_lower) -> person_id   供快速别名匹配
        self._alias_cache: TTLCache[Tuple[str, str], str] = TTLCache(
            maxsize=500, ttl=Config.RELATIONSHIP_ALIAS_TTL_SEC
        )

    # ============== public API ==============
    def list_active(self, user_id: str) -> List[Dict[str, Any]]:
        """取某用户所有正常(status=1)人物。"""
        if user_id not in self._persons_lru:
            self._reload_persons(user_id)
        return [dict(d) for d in self._persons_lru[user_id].values()]

    def get_person(self, user_id: str, person_id: str) -> Optional[Dict[str, Any]]:
        if user_id not in self._persons_lru:
            self._reload_persons(user_id)
        p = self._persons_lru[user_id].get(person_id)
        return dict(p) if p else None

    def find_related(
            self,
            user_id: str,
            text: str,
    ) -> List[Dict[str, Any]]:
        """根据文本匹配当前提到的人物。

        仅返回当前文本中明确出现的 canonical_name 或 alias。
        不会返回全部人物。
        """
        if not text or not text.strip():
            return []

        text_l = text.lower()

        if user_id not in self._persons_lru:
            self._reload_persons(user_id)

        persons = self._persons_lru[user_id].values()

        matched: Dict[str, Dict[str, Any]] = {}

        for person in persons:
            person_id = person.get("person_id")
            canonical_name = person.get("canonical_name") or ""

            # 主名匹配
            if canonical_name and canonical_name.lower() in text_l:
                matched[person_id] = dict(person)
                continue

            # 别名匹配
            aliases = person.get("aliases_json") or []

            if isinstance(aliases, str):
                aliases = _parse_json(aliases) or []

            for alias in aliases:
                if not isinstance(alias, str):
                    continue

                alias = alias.strip()

                if alias and alias.lower() in text_l:
                    matched[person_id] = dict(person)
                    break

        return list(matched.values())

    def resolve_name(self, user_id: str, name: str) -> Optional[Dict[str, Any]]:
        """给定名字（可能是别名/主名），反查人物。命中返回 row_dict，否则 None。"""
        if not name:
            return None
        name_l = name.strip().lower()

        # 1) alias TTL 缓存
        cache_key = (user_id, name_l)
        if cache_key in self._alias_cache:
            pid = self._alias_cache[cache_key]
            p = self.get_person(user_id, pid)
            if p:
                return p
            self._alias_cache.pop(cache_key, None)

        # 2) canonical_name 精确匹配（大小写不敏感）
        all_p = self.list_active(user_id)
        for p in all_p:
            if p.get("canonical_name", "").lower() == name_l:
                self._alias_cache[cache_key] = p["person_id"]
                return dict(p)

        # 3) aliases_json 精确匹配
        for p in all_p:
            aliases = p.get("aliases_json") or []
            for a in aliases:
                if isinstance(a, str) and a.lower() == name_l:
                    self._alias_cache[cache_key] = p["person_id"]
                    return dict(p)

        # 4) DB 兜底模糊 LIKE（极少量数据不care性能；找不到返回None）
        row = self._db_search_like(user_id, name)
        if row is not None:
            self._alias_cache[cache_key] = row["person_id"]
            # 写回 LRU（如果当前LRU存在）
            if user_id in self._persons_lru and row["status"] == PERSON_STATUS_ACTIVE:
                self._persons_lru[user_id][row["person_id"]] = row
            return row
        return None

    def add_or_update(
        self,
        user_id: str,
        canonical_name: str,
        aliases: Optional[Iterable[str]] = None,
        relation: str = "unknown",
        extra: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        person_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """新增或更新一个人物。若传 person_id 则按其更新（用于写入已识别人物的新别名）。"""
        if not canonical_name:
            raise ValueError("[Relationship] canonical_name 不能为空")

        if person_id is None:
            # 先尝试用 canonical_name 反查已有的，避免重复建
            existing = self.resolve_name(user_id, canonical_name)
            if existing is not None:
                person_id = existing["person_id"]

        if person_id is None:
            person_id = f"person_{uuid.uuid4().hex[:8]}"

        aliases_list: List[str] = list(dict.fromkeys(filter(None, aliases or [])))
        extra_dict: Dict[str, Any] = extra or {}
        now = _dt.datetime.now()

        sql = text(
            """
            INSERT INTO relationships
                (user_id, person_id, canonical_name, aliases_json, relation, extra_json, confidence, status, created_at, updated_at)
            VALUES
                (:uid, :pid, :cname, :aj, :rel, :ej, :conf, 1, :now, :now)
            ON DUPLICATE KEY UPDATE
                canonical_name = IF(VALUES(canonical_name) IS NULL, canonical_name, VALUES(canonical_name)),
                aliases_json = JSON_MERGE_PRESERVE(aliases_json, VALUES(aliases_json)),
                relation = IF(VALUES(relation) = 'unknown' AND relation != 'unknown', relation, VALUES(relation)),
                extra_json = JSON_MERGE_PATCH(extra_json, VALUES(extra_json)),
                confidence = GREATEST(confidence, VALUES(confidence)),
                status = IF(status = 2, 2, IF(VALUES(status)=3, 3, 1)),
                updated_at = VALUES(updated_at)
            """
        )
        with get_db_session() as sess:
            sess.execute(
                sql,
                {
                    "uid": user_id,
                    "pid": person_id,
                    "cname": canonical_name,
                    "aj": json.dumps(aliases_list, ensure_ascii=False),
                    "rel": relation,
                    "ej": json.dumps(extra_dict, ensure_ascii=False),
                    "conf": round(float(confidence), 3),
                    "now": now,
                },
            )
        # 别名去重（JSON_MERGE_PRESERVE 可能重复）
        self._dedupe_aliases(user_id, person_id)
        # 清缓存
        self._persons_lru.pop(user_id, None)
        self._flush_alias_cache(user_id)
        # 返回最新数据
        return self._db_select_person(user_id, person_id) or {}

    def merge_person(self, user_id: str, source_person_id: str, target_person_id: str) -> None:
        """合并人物：source -> target。source标记为已合并，别名合并进target，episodic 的 person_ids 指向 target。"""
        if source_person_id == target_person_id:
            return
        now = _dt.datetime.now()
        with get_db_session() as sess:
            # 1) 拿 source 的别名 + relation/extra 信息
            src = sess.execute(
                text(
                    "SELECT aliases_json, relation, extra_json, confidence "
                    "FROM relationships WHERE user_id=:uid AND person_id=:pid FOR UPDATE"
                ),
                {"uid": user_id, "pid": source_person_id},
            ).mappings().first()
            if src is None:
                logger.warning(f"[Relationship] merge 找不到源人物 {source_person_id}")
                return
            # 2) target 做 upsert（别名/extra 合并，取更高置信度）
            sess.execute(
                text(
                    """
                    UPDATE relationships
                    SET aliases_json = JSON_MERGE_PRESERVE(aliases_json, :src_aj),
                        extra_json   = JSON_MERGE_PATCH(extra_json, :src_ej),
                        confidence   = GREATEST(confidence, :src_conf),
                        updated_at   = :now
                    WHERE user_id=:uid AND person_id=:tpid
                    """
                ),
                {
                    "uid": user_id,
                    "tpid": target_person_id,
                    "src_aj": _parse_json(src["aliases_json"]) or [],
                    "src_ej": _parse_json(src["extra_json"]) or {},
                    "src_conf": float(src["confidence"]),
                    "now": now,
                },
            )
            # 3) source 标记为已合并
            sess.execute(
                text(
                    "UPDATE relationships SET status=2, merged_into=:tpid, updated_at=:now "
                    "WHERE user_id=:uid AND person_id=:spid"
                ),
                {"uid": user_id, "tpid": target_person_id, "spid": source_person_id, "now": now},
            )
            # 4) episodic 中引用了 source 的 person_ids_json 替换成 target
            sess.execute(
                text(
                    """
                    UPDATE episodic
                    SET person_ids_json = JSON_ARRAY_REPLACE(person_ids_json, :spid, :tpid)
                    WHERE user_id = :uid
                      AND JSON_CONTAINS(person_ids_json, JSON_QUOTE(:spid))
                    """
                ),
                {"uid": user_id, "spid": source_person_id, "tpid": target_person_id},
            )
        self._dedupe_aliases(user_id, target_person_id)
        self._persons_lru.pop(user_id, None)
        self._flush_alias_cache(user_id)
        logger.info(f"[Relationship] 合并 {source_person_id} -> {target_person_id}")

    def clear(self, user_id: str) -> None:
        with get_db_session() as sess:
            sess.execute(text("DELETE FROM relationships WHERE user_id = :uid"), {"uid": user_id})
        self._persons_lru.pop(user_id, None)
        self._flush_alias_cache(user_id)

    # ============== internal ==============
    def _reload_persons(self, user_id: str) -> None:
        with get_db_session() as sess:
            rows = sess.execute(
                text(
                    "SELECT * FROM relationships WHERE user_id=:uid AND status = 1"
                ),
                {"uid": user_id},
            ).mappings().all()
        bucket: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            d = _row_to_dict(r)
            bucket[d["person_id"]] = d
        self._persons_lru[user_id] = bucket

    def _dedupe_aliases(self, user_id: str, person_id: str) -> None:
        with get_db_session() as sess:
            row = sess.execute(
                text("SELECT aliases_json FROM relationships WHERE user_id=:uid AND person_id=:pid"),
                {"uid": user_id, "pid": person_id},
            ).scalar()
        if not row:
            return
        aliases = _parse_json(row) or []
        if not isinstance(aliases, list):
            aliases = []
        seen = set()
        deduped = []
        for a in aliases:
            if not isinstance(a, str):
                continue
            k = a.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            deduped.append(a)
        new_json = json.dumps(deduped, ensure_ascii=False)
        with get_db_session() as sess:
            sess.execute(
                text(
                    "UPDATE relationships SET aliases_json = :aj, updated_at=NOW() "
                    "WHERE user_id=:uid AND person_id=:pid"
                ),
                {"aj": new_json, "uid": user_id, "pid": person_id},
            )

    def _flush_alias_cache(self, user_id: str) -> None:
        keys_to_drop = [k for k in self._alias_cache if k[0] == user_id]
        for k in keys_to_drop:
            self._alias_cache.pop(k, None)

    def _db_search_like(self, user_id: str, name: str) -> Optional[Dict[str, Any]]:
        like = f"%{name}%"
        with get_db_session() as sess:
            r = sess.execute(
                text(
                    """
                    SELECT * FROM relationships
                    WHERE user_id = :uid AND status = 1
                      AND (canonical_name LIKE :like
                           OR JSON_SEARCH(aliases_json, 'one', :exact) IS NOT NULL)
                    ORDER BY confidence DESC
                    LIMIT 1
                    """
                ),
                {"uid": user_id, "like": like, "exact": name},
            ).mappings().first()
        return _row_to_dict(r) if r else None

    def _db_select_person(self, user_id: str, person_id: str) -> Optional[Dict[str, Any]]:
        with get_db_session() as sess:
            r = sess.execute(
                text(
                    "SELECT * FROM relationships WHERE user_id=:uid AND person_id=:pid LIMIT 1"
                ),
                {"uid": user_id, "pid": person_id},
            ).mappings().first()
        return _row_to_dict(r) if r else None
