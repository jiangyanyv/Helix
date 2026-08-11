from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class VectorHit:
    """向量检索命中结果。"""

    id: int
    score: float
    payload: Dict[str, Any]

    @property
    def user_id(self) -> Optional[str]:
        """从 payload 获取 user_id。"""
        value = self.payload.get("user_id")
        return str(value) if value is not None else None


class VectorStore(ABC):
    """向量存储抽象。

    当前主要服务于 Episodic Memory。

    设计原则：
    - VectorStore 只负责向量库操作
    - 不负责 MySQL
    - 不负责 Embedding
    - 不负责 Memory 业务逻辑
    """

    @abstractmethod
    def ensure_collection(
        self,
        collection: str,
        dimension: int,
    ) -> None:
        """确保 collection 存在且维度正确。"""

    @abstractmethod
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
        """批量写入 / 更新向量。

        items:
            [
                (
                    point_id,
                    vector,
                    payload,
                ),
                ...
            ]
        """

    @abstractmethod
    def delete_points(
        self,
        collection: str,
        ids: Iterable[int],
    ) -> None:
        """按照 Point ID 批量删除。"""

    @abstractmethod
    def delete_by_filter(
        self,
        collection: str,
        filter_cond: Dict[str, Any],
    ) -> None:
        """按照 Payload Filter 删除 Points。

        例如：

        {
            "must": [
                {
                    "key": "user_id",
                    "match": {
                        "value": "user_001"
                    }
                }
            ]
        }
        """

    @abstractmethod
    def update_payload(
        self,
        collection: str,
        payload: Dict[str, Any],
        ids: Optional[Iterable[int]] = None,
        filter_cond: Optional[Dict[str, Any]] = None,
    ) -> None:
        """更新已有 Point 的 payload。

        ids 与 filter_cond 至少提供一个。
        """

    @abstractmethod
    def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        limit: int,
        filter_cond: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[VectorHit]:
        """向量相似度检索。

        filter_cond：
            Qdrant Filter dict。

        例如：

        {
            "must": [
                {
                    "key": "user_id",
                    "match": {
                        "value": "user_001"
                    }
                }
            ]
        }
        """

    @abstractmethod
    def close(self) -> None:
        """释放底层连接。"""