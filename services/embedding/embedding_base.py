from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Embedding 抽象基类。

    后续扩展本地模型/其他 API 只需新增子类。
    """

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """单文本转向量。失败应抛出异常，由调用方处理降级。"""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量转向量。返回顺序与输入严格一致。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度，建 Qdrant collection / FAISS 索引必须一致。"""

    @abstractmethod
    def close(self) -> None:
        """释放底层资源。"""