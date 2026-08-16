"""Tavily Search 工具。

为 task_agent 提供网络搜索能力。
Tavily 是专为 AI agent 设计的搜索 API，返回结构化结果。

使用前需要设置环境变量 TAVILY_KEY。
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from config import Config


class TavilySearch:
    """Tavily 搜索工具。"""

    def __init__(self) -> None:
        self._api_key = Config.TAVILY_KEY
        self._client = None

        if not self._api_key:
            logger.warning(
                "[TavilySearch] TAVILY_KEY 未配置，"
                "搜索功能不可用"
            )
            return

        try:
            from tavily import TavilyClient

            self._client = TavilyClient(
                api_key=self._api_key
            )
            logger.info("[TavilySearch] 初始化成功")

        except ImportError:
            logger.warning(
                "[TavilySearch] tavily 包未安装，"
                "请运行 pip install tavily-python"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"[TavilySearch] 初始化失败: {e}"
            )

    @property
    def available(self) -> bool:
        """工具是否可用。"""

        return self._client is not None

    def search(
        self,
        query: str,
        max_results: int = 3,
    ) -> str:
        """执行搜索，返回格式化文本。

        返回格式：
            - [标题1] 内容摘要... (来源: url)
            - [标题2] 内容摘要... (来源: url)

        失败时返回空字符串，由调用方降级处理。
        """

        if not self.available:
            logger.warning(
                "[TavilySearch] 工具不可用，返回空"
            )
            return ""

        if not query or not query.strip():
            return ""

        try:
            response = self._client.search(
                query=query.strip(),
                max_results=max_results,
                search_depth="basic",
            )

            results = response.get("results", [])

            if not results:
                logger.info(
                    f"[TavilySearch] 无搜索结果 | query={query}"
                )
                return ""

            lines = []

            for r in results:
                title = r.get("title", "")
                content = r.get("content", "")
                url = r.get("url", "")

                # 截取内容摘要（避免过长）
                if len(content) > 200:
                    content = content[:200] + "..."

                lines.append(
                    f"- [{title}] {content}"
                    + (f" (来源: {url})" if url else "")
                )

            result_text = "\n".join(lines)

            logger.info(
                f"[TavilySearch] 搜索完成 | "
                f"query={query} | "
                f"results={len(results)}"
            )

            return result_text

        except Exception as e:  # noqa: BLE001
            logger.exception(
                f"[TavilySearch] 搜索失败 | "
                f"query={query}, error={e}"
            )
            return ""


# 全局单例
tavily_search = TavilySearch()
