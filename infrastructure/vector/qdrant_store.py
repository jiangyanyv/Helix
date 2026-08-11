from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger
from qdrant_client import QdrantClient, models

from config import Config
from .vector_base import VectorHit, VectorStore


class QdrantVectorStore(VectorStore):
    """Qdrant VectorStore 实现。

    当前适配：
        qdrant-client==1.12.2

    职责：
    - Collection 创建 / 校验
    - Payload Index 创建
    - Vector Upsert
    - Point 删除
    - Filter 删除
    - Payload 更新
    - Vector Search
    - Client 生命周期管理

    注意：
        Qdrant 只是向量索引。
        MySQL 才是 Episodic 的事实数据源。
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        api_key: Optional[str] = None,
        prefer_grpc: Optional[bool] = None,
    ) -> None:

        # ----------------------------------------------------
        # 使用 is None，而不是 or
        # ----------------------------------------------------

        host = (
            Config.QDRANT_HOST
            if host is None
            else host
        )

        port = (
            Config.QDRANT_PORT
            if port is None
            else port
        )

        grpc_port = (
            Config.QDRANT_GRPC_PORT
            if grpc_port is None
            else grpc_port
        )

        api_key = (
            Config.QDRANT_API_KEY
            if api_key is None
            else api_key
        )

        prefer_grpc = (
            Config.QDRANT_USE_GRPC
            if prefer_grpc is None
            else prefer_grpc
        )

        kwargs: Dict[str, Any] = {
            "host": host,
            "prefer_grpc": prefer_grpc,
        }

        if prefer_grpc:
            kwargs["grpc_port"] = grpc_port
        else:
            kwargs["port"] = port

        if api_key:
            kwargs["api_key"] = api_key

        self._client = QdrantClient(**kwargs)

        logger.info(
            f"[Qdrant] client initialized "
            f"host={host} "
            f"port={port} "
            f"grpc={prefer_grpc}"
        )

    # ========================================================
    # Public
    # ========================================================

    @property
    def client(self) -> QdrantClient:
        """暴露底层 QdrantClient。"""
        return self._client

    # ========================================================
    # Collection
    # ========================================================

    def ensure_collection(
        self,
        collection: str,
        dimension: int,
    ) -> None:
        """确保 collection 存在且 dimension 正确。"""

        if not collection:
            raise ValueError(
                "[Qdrant] collection 不能为空"
            )

        if dimension <= 0:
            raise ValueError(
                f"[Qdrant] dimension 必须 > 0，"
                f"收到 {dimension}"
            )

        # ----------------------------------------------------
        # 先检查 Collection 是否存在
        #
        # qdrant-client 1.12.2 支持 collection_exists。
        # ----------------------------------------------------

        try:
            exists = self._client.collection_exists(
                collection_name=collection
            )
        except Exception as exc:
            logger.exception(
                f"[Qdrant] 检查 collection 失败："
                f"{collection}"
            )
            raise RuntimeError(
                f"[Qdrant] 无法检查 collection={collection}"
            ) from exc

        # ----------------------------------------------------
        # 已存在
        # ----------------------------------------------------

        if exists:
            try:
                info = self._client.get_collection(
                    collection_name=collection
                )
            except Exception as exc:
                logger.exception(
                    f"[Qdrant] 获取 collection 信息失败："
                    f"{collection}"
                )
                raise RuntimeError(
                    f"[Qdrant] 无法获取 collection={collection}"
                ) from exc

            try:
                vectors_config = info.config.params.vectors
                actual_dimension = vectors_config.size
            except Exception as exc:
                raise RuntimeError(
                    f"[Qdrant] 无法读取 collection="
                    f"{collection} 的 vector config"
                ) from exc

            if actual_dimension != dimension:
                raise RuntimeError(
                    f"[Qdrant] collection={collection} "
                    f"dimension 不匹配："
                    f"期望={dimension}，"
                    f"实际={actual_dimension}"
                )

            logger.debug(
                f"[Qdrant] collection 已存在："
                f"{collection}, dim={dimension}"
            )

            # 即使 Collection 已存在，
            # 也确保 Payload Index 存在。
            self._ensure_payload_indexes(
                collection
            )

            return

        # ----------------------------------------------------
        # 创建 Collection
        # ----------------------------------------------------

        try:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
            )
        except Exception as exc:
            logger.exception(
                f"[Qdrant] 创建 collection 失败："
                f"{collection}"
            )
            raise RuntimeError(
                f"[Qdrant] 创建 collection={collection} 失败"
            ) from exc

        self._ensure_payload_indexes(
            collection
        )

        logger.info(
            f"[Qdrant] 创建 collection="
            f"{collection}, dim={dimension}"
        )

    def _ensure_payload_indexes(
        self,
        collection: str,
    ) -> None:
        """确保 Episodic 所需 Payload Index 存在。"""

        indexes = [
            (
                "user_id",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "tags",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "person_ids",
                models.PayloadSchemaType.KEYWORD,
            ),
            (
                "importance",
                models.PayloadSchemaType.FLOAT,
            ),
            (
                "timestamp",
                models.PayloadSchemaType.FLOAT,
            ),
        ]

        for field_name, field_schema in indexes:
            try:
                self._client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema=field_schema,
                )

                # logger.debug(
                #     f"[Qdrant] payload index 创建成功："
                #     f"{collection}.{field_name}"
                # )

            except Exception as exc:
                # 已经存在时，Qdrant 可能抛异常。
                # 这里不影响主流程。
                logger.debug(
                    f"[Qdrant] payload index "
                    f"{collection}.{field_name} "
                    f"创建/检查失败：{exc}"
                )

    # ========================================================
    # Upsert
    # ========================================================

    def upsert_points(
        self,
        collection: str,
        items: Sequence[
            Tuple[
                int,
                Sequence[float],
                Dict[str, Any],
            ]
        ],
    ) -> None:
        """批量 upsert points。"""

        if not items:
            return

        batch_ids: List[int] = []
        batch_vectors: List[List[float]] = []
        batch_payloads: List[Dict[str, Any]] = []

        for point_id, vector, payload in items:

            if not vector:
                raise ValueError(
                    f"[Qdrant] point={point_id} "
                    f"vector 不能为空"
                )

            vector_list = [
                float(value)
                for value in vector
            ]

            batch_ids.append(
                int(point_id)
            )

            batch_vectors.append(
                vector_list
            )

            batch_payloads.append(
                dict(payload)
            )

        try:
            self._client.upsert(
                collection_name=collection,
                points=models.Batch(
                    ids=batch_ids,
                    vectors=batch_vectors,
                    payloads=batch_payloads,
                ),
            )
        except Exception as exc:
            logger.exception(
                f"[Qdrant] upsert 失败："
                f"collection={collection}, "
                f"count={len(items)}"
            )
            raise RuntimeError(
                "[Qdrant] upsert_points 失败"
            ) from exc

    # ========================================================
    # Delete By IDs
    # ========================================================

    def delete_points(
        self,
        collection: str,
        ids: Iterable[int],
    ) -> None:
        """按照 Point ID 批量删除。"""

        ids_list = [
            int(point_id)
            for point_id in ids
        ]

        if not ids_list:
            return

        try:
            self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(
                    points=ids_list
                ),
            )
        except Exception as exc:
            logger.exception(
                f"[Qdrant] delete_points 失败："
                f"collection={collection}, "
                f"count={len(ids_list)}"
            )
            raise RuntimeError(
                "[Qdrant] delete_points 失败"
            ) from exc

    # ========================================================
    # Delete By Filter
    # ========================================================

    def delete_by_filter(
        self,
        collection: str,
        filter_cond: Dict[str, Any],
    ) -> None:
        """按照 Payload Filter 删除 Points。"""

        if not filter_cond:
            raise ValueError(
                "[Qdrant] filter_cond 不能为空"
            )

        qd_filter = models.Filter(
            **filter_cond
        )

        try:
            self._client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=qd_filter
                ),
            )
        except Exception as exc:
            logger.exception(
                f"[Qdrant] delete_by_filter 失败："
                f"collection={collection}"
            )
            raise RuntimeError(
                "[Qdrant] delete_by_filter 失败"
            ) from exc

    # ========================================================
    # Update Payload
    # ========================================================

    def update_payload(
        self,
        collection: str,
        payload: Dict[str, Any],
        ids: Optional[Iterable[int]] = None,
        filter_cond: Optional[Dict[str, Any]] = None,
    ) -> None:
        """更新 Point Payload。

        ids 与 filter_cond 至少提供一个。
        """

        if not payload:
            return

        if ids is None and not filter_cond:
            raise ValueError(
                "[Qdrant] update_payload 必须提供 "
                "ids 或 filter_cond"
            )

        # ----------------------------------------------------
        # Point ID selector
        # ----------------------------------------------------

        if ids is not None:

            ids_list = [
                int(point_id)
                for point_id in ids
            ]

            if not ids_list:
                return

            points_selector = (
                models.PointIdsList(
                    points=ids_list
                )
            )

        # ----------------------------------------------------
        # Filter selector
        # ----------------------------------------------------

        else:
            qd_filter = models.Filter(
                **filter_cond
            )

            points_selector = (
                models.FilterSelector(
                    filter=qd_filter
                )
            )

        try:
            self._client.set_payload(
                collection_name=collection,
                payload=dict(payload),
                points=points_selector,
            )
        except Exception as exc:
            logger.exception(
                f"[Qdrant] update_payload 失败："
                f"collection={collection}"
            )
            raise RuntimeError(
                "[Qdrant] update_payload 失败"
            ) from exc

    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        limit: int,
        filter_cond: Optional[
            Dict[str, Any]
        ] = None,
        score_threshold: Optional[float] = None,
    ) -> List[VectorHit]:
        """向量相似度搜索。

        适配：
            qdrant-client==1.12.2

        注意：
            Qdrant 搜索失败时抛异常，
            由 EpisodicService 负责 fallback。
        """

        if not query_vector:
            return []

        if limit <= 0:
            return []

        qvec = [
            float(value)
            for value in query_vector
        ]

        qd_filter = None

        if filter_cond:
            qd_filter = models.Filter(
                **filter_cond
            )

        try:
            points = self._client.search(
                collection_name=collection,
                query_vector=qvec,
                limit=int(limit),
                query_filter=qd_filter,
                score_threshold=(
                    float(score_threshold)
                    if score_threshold is not None
                    else None
                ),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.exception(
                f"[Qdrant] search 失败："
                f"collection={collection}"
            )

            # 不返回 []。
            #
            # [] 表示：
            # 「搜索成功，但是没有命中」
            #
            # Exception 才表示：
            # 「Qdrant 搜索失败」
            #
            # 让 EpisodicService 决定 fallback。
            raise RuntimeError(
                "[Qdrant] search 失败"
            ) from exc

        hits: List[VectorHit] = []

        for point in points:
            hits.append(
                VectorHit(
                    id=int(point.id),
                    score=float(
                        getattr(
                            point,
                            "score",
                            0.0,
                        )
                    ),
                    payload=dict(
                        point.payload or {}
                    ),
                )
            )

        return hits

    # ========================================================
    # Close
    # ========================================================

    def close(self) -> None:
        """关闭 Qdrant Client。"""

        try:
            self._client.close()

            logger.info(
                "[Qdrant] client closed"
            )

        except Exception as exc:
            logger.warning(
                f"[Qdrant] client close 失败：{exc}"
            )