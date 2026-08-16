"""Planner Builder：纯逻辑构建 Planner 请求 + 解析结果。

职责：
    1. build(user_input) → ChatRequest（构建 LLM 判断请求）
    2. parse_result(raw) → {need_tool, tool_name}（解析 LLM 输出）

不包含：
    - LLM 调用（由 Node 层执行 container.llm_client.generate）
    - 副作用（如放过渡回复到 AudioQueue，在 planner_node 内做）

这样设计便于：
    - 独立测试（无需起 LangGraph / Redis）
    - 切换判断策略（LLM → 规则、单 prompt → 多 prompt），不影响 Node 层
    - 统一 prompt 格式
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from services.llm.chat_request import ChatRequest


# ============================================================
# Prompt
# ============================================================

PLANNER_SYSTEM_PROMPT = """你是一个意图判断器。判断用户输入是否需要联网搜索实时信息。

需要搜索的：
- 天气、新闻、最新数据、人物近况、实时价格、赛事比分
- 任何你无法从自身知识回答的时效性问题

不需要搜索的：
- 闲聊、情绪倾诉、自我介绍、知识问答、意见建议
- 用户在分享个人事情、经历、感受

跨轮次意图识别：
- 如果提供了上文对话，用户本轮消息可能省略主题（如"那上海呢"、"再来一个"、"还有呢"）
- 此时应结合上文判断：上文在查实时信息且本轮是对同主题的追问/延续 → need_tool=true
- 上文已在闲聊且本轮明显切换话题 → 按本轮内容判断

只返回 JSON：{"need_tool": true} 或 {"need_tool": false}
不要返回任何其他内容。"""


# ============================================================
# DTO
# ============================================================

@dataclass
class PlannerDecision:
    """Planner 决策结果。"""

    need_tool: bool
    tool_name: str | None


# ============================================================
# Builder
# ============================================================

class PlannerBuilder:
    """构建 Planner 的 LLM 请求 + 解析返回。"""

    def __init__(self, model_name: str) -> None:
        """
        Args:
            model_name: LLM 模型名（如 "deepseek-chat"）。
        """

        self._model_name = model_name

    # ------------------------------------------------------
    # 构建请求
    # ------------------------------------------------------

    def build(
        self,
        user_input: str,
        recent_history: list | None = None,
    ) -> ChatRequest:
        """根据用户输入构建 LLM 请求。

        Args:
            user_input: 用户原始输入文本
            recent_history: 最近 N 轮对话消息（List[BaseMessage]）。
                传入后 Planner 可结合上文判断省略句的意图，
                None 时仅看当前这句（与原行为一致）。

        Returns:
            可直接传给 llm_client.generate() 的 ChatRequest
        """

        messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT)]

        if recent_history:
            messages.extend(recent_history)

        messages.append(HumanMessage(content=user_input))

        return ChatRequest(
            messages=messages,
            model=self._model_name,
            stream=False,
            temperature=0.0,
            top_p=1.0,
            max_tokens=20,
        )

    # ------------------------------------------------------
    # 解析结果
    # ------------------------------------------------------

    def parse_result(self, raw: str) -> PlannerDecision:
        """解析 LLM 返回的原始文本，得到 PlannerDecision。

        支持多种 LLM 返回格式：
        - {"need_tool": true}
        - true
        - 文本中包含 true

        Args:
            raw: LLM generate() 返回的 text

        Returns:
            PlannerDecision(need_tool, tool_name)
            need_tool=True 时默认使用 "tavily_search"
        """

        need_tool = self._parse_need_tool(raw)
        tool_name = "tavily_search" if need_tool else None

        return PlannerDecision(
            need_tool=need_tool,
            tool_name=tool_name,
        )

    # ------------------------------------------------------
    # Internal: need_tool 解析
    # ------------------------------------------------------

    @staticmethod
    def _parse_need_tool(raw: str) -> bool:
        if not raw:
            return False

        # JSON 解析
        try:
            data = json.loads(raw)

            if isinstance(data, dict):
                return bool(data.get("need_tool", False))

            if isinstance(data, bool):
                return data

        except (json.JSONDecodeError, TypeError):
            pass

        # 兜底：文本中找 true/false
        raw_lower = raw.lower().strip()

        if "true" in raw_lower:
            return True

        return False
