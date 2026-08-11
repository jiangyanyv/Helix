from __future__ import annotations

import datetime as _dt
import json
import time
from typing import (TYPE_CHECKING,Any,Dict,Iterable,List,Optional,Sequence,Tuple,)

from cachetools import TTLCache
from loguru import logger
from sqlalchemy import text

from config import Config
from infrastructure.database.mysql import get_db_session

if TYPE_CHECKING:
    from services.embedding.embedding_base import EmbeddingProvider
    from infrastructure.vector.vector_base import VectorStore


# ============================================================
# Constants
# ============================================================

DEFAULT_IMPORTANCE = 0.5

MAX_TOP_K = Config.MAX_EPISODIC_TOP_K


# ============================================================
# Helper Functions
# ============================================================

def _clamp_importance(
    value: float,
) -> float:
    """将 importance 限制在 [0, 1]。"""

    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[Episodic] importance 必须是数字，"
            f"收到 {value!r}"
        ) from exc

    return max(
        0.0,
        min(1.0, value),
    )


def _normalize_top_k(
    top_k: int,
) -> int:
    """规范化 top_k。"""

    try:
        top_k = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[Episodic] top_k 必须是整数，"
            f"收到 {top_k!r}"
        ) from exc

    return max(
        1,
        min(top_k, MAX_TOP_K),
    )


def _loads(value: Any) -> Any:
    """解析 JSON 字段。"""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(
                f"[Episodic] JSON 解析失败："
                f"{value[:100]!r}"
            )
            return value

    return value


def _normalize_json_list(
    value: Any,
) -> List[Any]:
    """将值规范化为 list。"""

    value = _loads(value)

    if value is None:
        return []

    if isinstance(value, list):
        return value

    logger.warning(
        f"[Episodic] JSON list 类型异常："
        f"{type(value).__name__}"
    )

    return []


def _normalize_json_dict(
    value: Any,
) -> Dict[str, Any]:
    """将值规范化为 dict。"""

    value = _loads(value)

    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    logger.warning(
        f"[Episodic] JSON dict 类型异常："
        f"{type(value).__name__}"
    )

    return {}


def _row_to_event(
    row: Any,
) -> Dict[str, Any]:
    """将 MySQL Row 转换为业务 Event。"""

    data = (
        dict(row._mapping)
        if hasattr(row, "_mapping")
        else dict(row)
    )

    tags = _normalize_json_list(
        data.get("tags_json")
    )

    person_ids = _normalize_json_list(
        data.get("person_ids_json")
    )

    metadata = _normalize_json_dict(
        data.get("metadata_json")
    )

    importance = _clamp_importance(
        data.get(
            "importance",
            DEFAULT_IMPORTANCE,
        )
    )

    return {
        "id": int(data["id"]),
        "user_id": data["user_id"],
        "content": data["content"],
        "tags": tags,
        "person_ids": person_ids,
        "metadata": metadata,
        "timestamp": data["timestamp"],
        "created_at": data["created_at"],
        "importance": importance,
    }


# ============================================================
# EpisodicService
# ============================================================

class EpisodicService:
    """情景记忆服务。

    数据源：

        MySQL
          ↓
        Source of Truth

    检索：

        Embedding
          ↓
        Qdrant
          ↓
        MySQL 回表

    降级：

        Qdrant 不可用
          ↓
        search_recent()

    缓存：

        TTLCache
          ↓
        查询结果缓存

    不 preload 全量事件。
    """

    def __init__(
        self,
        embedding_provider: Optional[
            "EmbeddingProvider"
        ] = None,
        vector_store: Optional[
            "VectorStore"
        ] = None,
    ) -> None:

        self._query_cache: TTLCache[
            Tuple[Any, ...],
            List[Dict[str, Any]],
        ] = TTLCache(
            maxsize=Config.EPISODIC_QUERY_CACHE_MAXSIZE,
            ttl=Config.EPISODIC_QUERY_CACHE_TTL_SEC,
            timer=time.monotonic,
        )

        self._embedding = embedding_provider
        self._vector_store = vector_store

        self._collection_ready = False

    # ========================================================
    # External Binding
    # ========================================================

    def bind_external(
        self,
        embedding_provider: "EmbeddingProvider",
        vector_store: "VectorStore",
    ) -> None:
        """注入 EmbeddingProvider + VectorStore。"""

        if embedding_provider is None:
            raise ValueError(
                "[Episodic] embedding_provider 不能为空"
            )

        if vector_store is None:
            raise ValueError(
                "[Episodic] vector_store 不能为空"
            )

        self._embedding = embedding_provider
        self._vector_store = vector_store

        self._collection_ready = False

        logger.info(
            "[Episodic] EmbeddingProvider + "
            "VectorStore 已绑定"
        )

    # ========================================================
    # Collection
    # ========================================================

    def ensure_collection_ready(self) -> bool:
        """确保 Episodic Collection 可用。"""

        if self._collection_ready:
            return True

        if (
            self._embedding is None
            or self._vector_store is None
        ):
            return False

        try:
            self._vector_store.ensure_collection(
                Config.QDRANT_COLLECTION_EPISODIC,
                self._embedding.dimension,
            )

            self._collection_ready = True

            return True

        except Exception as exc:
            self._collection_ready = False

            logger.warning(
                f"[Episodic] Qdrant collection "
                f"初始化失败：{exc}"
            )

            return False

    # ========================================================
    # Add
    # ========================================================

    def add(
        self,
        user_id: str,
        content: str,
        tags: Optional[Iterable[str]] = None,
        person_ids: Optional[Iterable[str]] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        timestamp: Optional[
            _dt.datetime
        ] = None,
        importance: float = DEFAULT_IMPORTANCE,
    ) -> int:
        """新增一条 Episodic。

        返回：
            MySQL 自增 ID。

        MySQL ID 同时作为 Qdrant Point ID。
        """

        self._validate_user_id(
            user_id
        )

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError(
                "[Episodic] content 不能为空"
            )

        content = content.strip()

        ts = timestamp or _dt.datetime.now()
        now = _dt.datetime.now()

        importance = _clamp_importance(
            importance
        )

        tags_list = self._normalize_string_list(
            tags
        )

        person_list = self._normalize_string_list(
            person_ids
        )

        meta_dict = (
            dict(metadata)
            if metadata is not None
            else {}
        )

        # ----------------------------------------------------
        # MySQL
        # ----------------------------------------------------

        sql = text(
            """
            INSERT INTO episodic
                (
                    user_id,
                    content,
                    tags_json,
                    person_ids_json,
                    metadata_json,
                    timestamp,
                    created_at,
                    importance
                )
            VALUES
                (
                    :uid,
                    :content,
                    :tags,
                    :pids,
                    :meta,
                    :ts,
                    :now,
                    :imp
                )
            """
        )

        with get_db_session() as sess:
            result = sess.execute(
                sql,
                {
                    "uid": user_id,
                    "content": content,
                    "tags": json.dumps(
                        tags_list,
                        ensure_ascii=False,
                    ),
                    "pids": json.dumps(
                        person_list,
                        ensure_ascii=False,
                    ),
                    "meta": json.dumps(
                        meta_dict,
                        ensure_ascii=False,
                    ),
                    "ts": ts,
                    "now": now,
                    "imp": importance,
                },
            )

            new_id = int(
                result.lastrowid or 0
            )

        if new_id <= 0:
            raise RuntimeError(
                "[Episodic] INSERT 未返回有效自增 id"
            )

        # 新增数据后，该用户的旧查询结果可能失效。
        # 只清除当前用户的缓存，避免影响其他用户的命中率。
        self._invalidate_user_cache(user_id)

        logger.info(
            f"[Episodic] add "
            f"id={new_id} "
            f"user_id={user_id} "
            f"len={len(content)}"
        )

        # ----------------------------------------------------
        # Qdrant
        # ----------------------------------------------------

        if not self.ensure_collection_ready():
            return new_id

        if (
            self._embedding is None
            or self._vector_store is None
        ):
            return new_id

        try:
            vector = self._embedding.embed(
                content
            )

            payload: Dict[str, Any] = {
                "user_id": user_id,
                "tags": tags_list,
                "person_ids": person_list,
                "importance": importance,
                "timestamp": ts.timestamp(),
            }

            self._vector_store.upsert_points(
                Config.QDRANT_COLLECTION_EPISODIC,
                [
                    (
                        new_id,
                        vector,
                        payload,
                    )
                ],
            )

        except Exception as exc:
            self._collection_ready = False

            logger.warning(
                f"[Episodic] Qdrant 写入失败 "
                f"id={new_id}：{exc}"
            )

            # MySQL 不回滚。
            #
            # MySQL 是 Source of Truth。
            # 后续可以通过 VectorSync 补偿。

        return new_id

    # ========================================================
    # Get By IDs
    # ========================================================

    def get_by_ids(
        self,
        user_id: str,
        ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """按照 ID 获取事件，并保持输入顺序。"""

        self._validate_user_id(
            user_id
        )

        if not ids:
            return []

        normalized_ids = list(
            dict.fromkeys(
                int(event_id)
                for event_id in ids
            )
        )

        placeholders = ",".join(
            f":i{index}"
            for index in range(
                len(normalized_ids)
            )
        )

        sql = text(
            f"""
            SELECT *
            FROM episodic
            WHERE user_id = :uid
              AND id IN ({placeholders})
            """
        )

        params: Dict[str, Any] = {
            "uid": user_id,
        }

        for index, event_id in enumerate(
            normalized_ids
        ):
            params[
                f"i{index}"
            ] = event_id

        with get_db_session() as sess:
            rows = sess.execute(
                sql,
                params,
            ).mappings().all()

        events = [
            _row_to_event(row)
            for row in rows
        ]

        id_order = {
            event_id: index
            for index, event_id in enumerate(
                normalized_ids
            )
        }

        events.sort(
            key=lambda event: id_order.get(
                event["id"],
                999999,
            )
        )

        return events

    # ========================================================
    # Search Recent
    # ========================================================

    def search_recent(
        self,
        user_id: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """最近事件检索。"""

        self._validate_user_id(
            user_id
        )

        top_k = _normalize_top_k(
            top_k
        )

        cache_key = (
            user_id,
            "__recent__",
            top_k,
        )

        cached = self._query_cache.get(
            cache_key
        )

        if cached is not None:
            return self._copy_events(
                cached
            )

        sql = text(
            """
            SELECT *
            FROM episodic
            WHERE user_id = :uid
            ORDER BY timestamp DESC, id DESC
            LIMIT :lim
            """
        )

        with get_db_session() as sess:
            rows = sess.execute(
                sql,
                {
                    "uid": user_id,
                    "lim": top_k,
                },
            ).mappings().all()

        result = [
            _row_to_event(row)
            for row in rows
        ]

        self._query_cache[cache_key] = (
            self._copy_events(result)
        )

        return result

    # ========================================================
    # Semantic Search
    # ========================================================

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[
            float
        ] = 0.3,
    ) -> List[Dict[str, Any]]:
        """语义检索。

        正常：

            query
              ↓
            embedding
              ↓
            Qdrant
              ↓
            user_id filter
              ↓
            MySQL 回表

        降级：

            Qdrant / Embedding 不可用
              ↓
            search_recent()

        返回的向量检索结果包含：
            _score
        """

        self._validate_user_id(
            user_id
        )

        top_k = _normalize_top_k(
            top_k
        )

        normalized_query = (
            query.strip()
            if isinstance(query, str)
            else ""
        )

        normalized_threshold = (
            None
            if score_threshold is None
            else float(score_threshold)
        )

        cache_key = (
            user_id,
            normalized_query,
            top_k,
            normalized_threshold,
        )

        cached = self._query_cache.get(
            cache_key
        )

        if cached is not None:
            return self._copy_events(
                cached
            )

        # ----------------------------------------------------
        # 空 Query
        # ----------------------------------------------------

        if not normalized_query:
            return self.search_recent(
                user_id,
                top_k,
            )

        # ----------------------------------------------------
        # Vector 不可用
        # ----------------------------------------------------

        if not self.ensure_collection_ready():
            result = self.search_recent(
                user_id,
                top_k,
            )

            self._query_cache[cache_key] = (
                self._copy_events(result)
            )

            return result

        if (
            self._embedding is None
            or self._vector_store is None
        ):
            result = self.search_recent(
                user_id,
                top_k,
            )

            self._query_cache[cache_key] = (
                self._copy_events(result)
            )

            return result

        # ----------------------------------------------------
        # Embedding + Qdrant
        # ----------------------------------------------------

        try:
            query_vector = self._embedding.embed(
                normalized_query
            )

            filter_cond = {
                "must": [
                    {
                        "key": "user_id",
                        "match": {
                            "value": user_id,
                        },
                    }
                ],
            }

            candidate_limit = min(
                top_k * 3,
                MAX_TOP_K * 3,
            )

            hits = self._vector_store.search(
                Config.QDRANT_COLLECTION_EPISODIC,
                query_vector,
                limit=candidate_limit,
                filter_cond=filter_cond,
                score_threshold=normalized_threshold,
            )

        except Exception as exc:
            self._collection_ready = False

            logger.warning(
                f"[Episodic] 向量检索失败，"
                f"降级最近 N 条：{exc}"
            )

            result = self.search_recent(
                user_id,
                top_k,
            )

            self._query_cache[cache_key] = (
                self._copy_events(result)
            )

            return result

        # ----------------------------------------------------
        # Qdrant 没有结果
        # ----------------------------------------------------

        if not hits:
            fallback_k = max(
                top_k // 2,
                1,
            )

            result = self.search_recent(
                user_id,
                fallback_k,
            )

            self._query_cache[cache_key] = (
                self._copy_events(result)
            )

            return result

        # ----------------------------------------------------
        # Qdrant → MySQL
        # ----------------------------------------------------

        ids: List[int] = []

        score_map: Dict[int, float] = {}

        for hit in hits:
            event_id = int(hit.id)

            ids.append(event_id)

            score_map[event_id] = float(
                hit.score
            )

        events = self.get_by_ids(
            user_id,
            ids,
        )

        # ----------------------------------------------------
        # 附加 Score
        # ----------------------------------------------------

        for event in events:
            event["_score"] = score_map.get(
                event["id"],
                0.0,
            )

        events.sort(
            key=lambda event: event.get(
                "_score",
                0.0,
            ),
            reverse=True,
        )

        events = events[:top_k]

        self._query_cache[cache_key] = (
            self._copy_events(events)
        )

        return events

    # ========================================================
    # Update Importance
    # ========================================================

    def update_importance(
        self,
        user_id: str,
        event_id: int,
        importance: float,
    ) -> None:
        """更新事件重要性。

        MySQL 是事实源。
        Qdrant payload 同步更新。
        """

        self._validate_user_id(
            user_id
        )

        importance = _clamp_importance(
            importance
        )

        sql = text(
            """
            UPDATE episodic
            SET importance = :imp
            WHERE id = :id
              AND user_id = :uid
            """
        )

        with get_db_session() as sess:
            result = sess.execute(
                sql,
                {
                    "imp": importance,
                    "id": int(event_id),
                    "uid": user_id,
                },
            )

        # 仅清除当前用户的缓存
        self._invalidate_user_cache(user_id)

        if result.rowcount == 0:
            logger.warning(
                f"[Episodic] 未找到事件："
                f"user_id={user_id}, "
                f"event_id={event_id}"
            )

            return

        # ----------------------------------------------------
        # 同步 Qdrant Payload
        # ----------------------------------------------------

        if (
            self._vector_store is None
            or not self.ensure_collection_ready()
        ):
            return

        try:
            self._vector_store.update_payload(
                Config.QDRANT_COLLECTION_EPISODIC,
                payload={
                    "importance": importance,
                },
                ids=[int(event_id)],
            )

        except Exception as exc:
            self._collection_ready = False

            logger.warning(
                f"[Episodic] Qdrant importance "
                f"同步失败："
                f"user_id={user_id}, "
                f"event_id={event_id}, "
                f"error={exc}"
            )

            # 不回滚 MySQL。

    # ========================================================
    # Clear
    # ========================================================

    def clear(
        self,
        user_id: str,
    ) -> None:
        """清空指定用户的所有 Episodic。

        MySQL：
            删除事实数据。

        Qdrant：
            删除对应 user_id 的向量。

        Cache：
            清空查询缓存。
        """

        self._validate_user_id(
            user_id
        )

        # ----------------------------------------------------
        # 1. MySQL
        # ----------------------------------------------------

        with get_db_session() as sess:
            sess.execute(
                text(
                    """
                    DELETE FROM episodic
                    WHERE user_id = :uid
                    """
                ),
                {
                    "uid": user_id,
                },
            )

        # ----------------------------------------------------
        # 2. Cache
        # ----------------------------------------------------

        self._query_cache.clear()

        # ----------------------------------------------------
        # 3. Qdrant
        # ----------------------------------------------------

        if (
            self._vector_store is None
            or not self.ensure_collection_ready()
        ):
            logger.info(
                f"[Episodic] 清空 user_id={user_id} "
                f"（Qdrant 未启用）"
            )
            return

        try:
            filter_cond = {
                "must": [
                    {
                        "key": "user_id",
                        "match": {
                            "value": user_id,
                        },
                    }
                ],
            }

            self._vector_store.delete_by_filter(
                Config.QDRANT_COLLECTION_EPISODIC,
                filter_cond,
            )

        except Exception as exc:
            self._collection_ready = False

            logger.warning(
                f"[Episodic] Qdrant 清理失败："
                f"user_id={user_id}, "
                f"error={exc}"
            )

            # MySQL 已经成功删除。
            # Qdrant 残留后续可以通过索引重建解决。

        logger.info(
            f"[Episodic] 清空 user_id={user_id}"
        )

    # ========================================================
    # Cache
    # ========================================================

    def clear_query_cache(self) -> None:
        """清空 Episodic 查询缓存。"""

        self._query_cache.clear()

        logger.debug(
            "[Episodic] 查询缓存已清空"
        )

    def _invalidate_user_cache(
        self,
        user_id: str,
    ) -> None:
        """仅清除指定用户的查询缓存。

        cache key 结构为 (user_id, query, top_k, threshold)，
        首位元素是 user_id。遍历删除该用户的所有 key，
        避免影响其他用户的缓存命中率。

        add / update_importance 时调用此方法，
        替代原先的 _query_cache.clear() 全量清空。
        """

        # 快照 keys，避免边遍历边删除
        keys_to_remove = [
            key
            for key in list(self._query_cache.keys())
            if key and key[0] == user_id
        ]

        for key in keys_to_remove:

            try:

                del self._query_cache[key]

            except KeyError:

                # TTL 过期导致已被自动清除，忽略
                pass

        if keys_to_remove:

            logger.debug(
                f"[Episodic] 用户缓存失效 "
                f"user_id={user_id} "
                f"cleared={len(keys_to_remove)}"
            )

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _validate_user_id(
        user_id: str,
    ) -> None:
        """校验 user_id。"""

        if not isinstance(user_id, str):
            raise ValueError(
                "[Episodic] user_id 必须是 str，"
                f"收到 {type(user_id).__name__}"
            )

        if not user_id.strip():
            raise ValueError(
                "[Episodic] user_id 不能为空"
            )

    @staticmethod
    def _normalize_string_list(
        values: Optional[
            Iterable[str]
        ],
    ) -> List[str]:
        """清洗字符串列表并去重。"""

        if values is None:
            return []

        result: List[str] = []
        seen = set()

        for value in values:
            if value is None:
                continue

            value = str(value).strip()

            if not value:
                continue

            if value in seen:
                continue

            seen.add(value)
            result.append(value)

        return result

    @staticmethod
    def _copy_events(
        events: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """复制 Cache 数据，避免调用方修改缓存。"""

        return [
            dict(event)
            for event in events
        ]