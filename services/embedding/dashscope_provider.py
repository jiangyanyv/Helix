from __future__ import annotations

import time
from typing import List, Optional

import httpx

from config import Config
from .embedding_base import EmbeddingProvider


class DashScopeEmbeddingProvider(EmbeddingProvider):
    """通义 DashScope Embedding API。

    支持 qwen3-text-embedding / qwen3.7-text-embedding 等模型。

    文档参考：
    https://help.aliyun.com/zh/model-studio/developer-reference/text-embedding-api
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or Config.DASHSCOPE_API_KEY

        if not self.api_key:
            raise RuntimeError(
                "[Embedding] DASHSCOPE_API_KEY 未配置。"
                "请在 .env 中填写"
            )

        self.model = model or Config.EMBEDDING_MODEL
        self._dimension = dimension or Config.EMBEDDING_DIM
        self.batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
        self.timeout = timeout or Config.EMBEDDING_TIMEOUT

        self._endpoint = (
            "https://dashscope.aliyuncs.com/api/v1/services/"
            "embeddings/text-embedding/text-embedding"
        )

        # 长生命周期复用 HTTP Client，避免每次请求重复创建连接。
        self._client = httpx.Client(
            timeout=self.timeout,
        )

        self._closed = False

    # ============== public ==============

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """单文本转向量。"""
        result = self.embed_batch([text])
        return result[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量转向量。

        DashScope 单次批量有限制，因此按照 batch_size
        分片发送，再按照输入顺序合并结果。
        """
        if not texts:
            return []

        self._ensure_open()

        outputs: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            outputs.extend(self._request(chunk))

        return outputs

    def close(self) -> None:
        """释放 HTTP Client。"""
        if not self._closed:
            self._client.close()
            self._closed = True

    # ============== internal ==============

    def _ensure_open(self) -> None:
        """确保 Provider 尚未关闭。"""
        if self._closed:
            raise RuntimeError(
                "[Embedding] DashScopeEmbeddingProvider 已关闭，"
                "无法继续发送请求"
            )

    def _request(self, texts: List[str]) -> List[List[float]]:
        """向 DashScope 发送一次 Embedding 请求。"""

        body = {
            "model": self.model,
            "input": {
                "texts": texts,
            },
            "parameters": {
                "text_type": "document",
                "dimension": self._dimension,
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None

        for attempt in range(3):
            try:
                self._ensure_open()

                # 复用 __init__ 创建的 HTTP Client
                resp = self._client.post(
                    self._endpoint,
                    json=body,
                    headers=headers,
                )

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"[Embedding DashScope] "
                        f"HTTP {resp.status_code} "
                        f"{resp.text[:300]}"
                    )

                data = resp.json()

                output = data.get("output", {}).get(
                    "embeddings",
                    []
                )

                if not output or len(output) != len(texts):
                    raise RuntimeError(
                        f"[Embedding DashScope] 返回长度不匹配: "
                        f"期望 {len(texts)} "
                        f"实际 {len(output)}. "
                        f"resp={str(data)[:400]}"
                    )

                # DashScope 返回带 index。
                # 按 index 排序，确保结果顺序与输入一致。
                sorted_items = sorted(
                    output,
                    key=lambda x: x.get("index", 0),
                )

                return [
                    item["embedding"]
                    for item in sorted_items
                ]

            except Exception as e:  # noqa: BLE001
                last_err = e

                # 最后一次失败后无需继续等待
                if attempt < 2:
                    time.sleep(1 + attempt)

        assert last_err is not None
        raise last_err