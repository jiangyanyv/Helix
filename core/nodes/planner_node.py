"""Planner Node：LangGraph Node wrapper。

职责（仅 Node 层职责）：
    1. 从 state 读取 user_input / turn_id
    2. 调用 Service 层 PlannerBuilder（纯逻辑）
       - build(user_input) → ChatRequest
       - LLM generate（通过 container.llm_client）
       - parse_result(raw) → PlannerDecision
    3. 副作用：need_tool=True 时放过渡回复 chunk 到 AudioQueue
       （Node 有操作 AudioQueue 的上下文，所以放这里）
    4. 返回决策结果给下一个节点

不包含：
    - Prompt 构建（在 planner_builder.py）
    - 解析逻辑（在 planner_builder.py）

符合项目约定：
    _node.py = LangGraph Node wrapper
    Service 文件 = 纯业务逻辑
"""

from __future__ import annotations

from loguru import logger
import random

from langchain_core.messages import SystemMessage

from core.state import AgentState
from services.container import container
from services.llm.stream_chunk import StreamChunk


# 喂给 Planner 的最近历史长度（条数）
# 4 条 = 最近 2 轮（2 H + 2 A），足以覆盖"上轮提到工具 → 本轮省略追问"的场景
PLANNER_RECENT_HISTORY_LIMIT = 4


# ============================================================
# 过渡回复模板
# ============================================================

TRANSITION_TEMPLATES = {
    "tavily_search": [
        "嗯嗯，我去搜搜看～",
        "好嘞，我翻翻资料库！",
        "等等哦，我找找看～",
        "查资料中，稍等我一下呀！",
        "我去瞄一眼，等等我呀～",
        "好哦，我帮你搜一下～",
        "等一下下，我去去就回！",
        "找东西我最拿手啦，稍等～",
    ],
}


# ============================================================
# Node
# ============================================================

def planner_node(state: AgentState):
    """Planner Node wrapper。

    通过 container.planner_builder 构建请求 + 解析结果，
    副作用（过渡回复）保留在 Node 层。
    """

    user_input = state.get("user_input", "")
    turn_id = state.get("turn_id", "")

    if not user_input or not user_input.strip():
        return {"need_tool": False, "tool_name": None}

    # 从 state.messages 过滤出最近 N 条对话历史
    # - 排除 SystemMessage（滚动摘要，对意图判断无价值且会污染 prompt）
    # - 失败时降级为只看当前 user_input（保持原行为）
    recent_history = _extract_recent_history(state)

    try:
        # Step1: 构建请求（Service 层纯逻辑）
        request = container.planner_builder.build(
            user_input,
            recent_history=recent_history,
        )

        # Step2: 执行 LLM 调用
        response = container.llm_client.generate(request)
        raw = (response.text or "").strip()

        # Step3: 解析结果（Service 层纯逻辑）
        decision = container.planner_builder.parse_result(raw)

        logger.info(
            f"[planner] need_tool={decision.need_tool} | "
            f"tool={decision.tool_name} | "
            f"input={user_input[:50]}"
        )

        # Step4: 副作用（仅 Node 层有上下文）
        if decision.need_tool and decision.tool_name:
            _send_transition(turn_id, decision.tool_name)

        return {
            "need_tool": decision.need_tool,
            "tool_name": decision.tool_name,
        }

    except Exception as e:  # noqa: BLE001
        logger.exception(
            f"[planner] LLM 判断失败，默认不调工具: {e}"
        )
        return {"need_tool": False, "tool_name": None}


# ============================================================
# Internal: 历史提取
# ============================================================

def _extract_recent_history(state: AgentState) -> list:
    """从 state.messages 过滤出最近 N 条对话历史。

    - 排除 SystemMessage（滚动摘要，对意图判断无价值且会污染 prompt）
    - 取最近 PLANNER_RECENT_HISTORY_LIMIT 条
    - 任何异常都降级为空列表（保持"只看当前 user_input"的原行为）

    Returns:
        List[BaseMessage]，可能为空
    """

    try:
        messages = state.get("messages") or []
        # 过滤掉 SystemMessage（主要是 ConversationManager 前置的摘要）
        filtered = [
            m for m in messages
            if not isinstance(m, SystemMessage)
        ]

        # 取最近 N 条
        if len(filtered) > PLANNER_RECENT_HISTORY_LIMIT:
            filtered = filtered[-PLANNER_RECENT_HISTORY_LIMIT:]

        return filtered

    except Exception as e:  # noqa: BLE001
        logger.debug(
            f"[planner] 提取历史失败，降级为只看当前输入: {e}"
        )
        return []


# ============================================================
# Internal: 副作用
# ============================================================

def _send_transition(turn_id: str, tool_name: str) -> None:
    """把过渡回复 chunk 放入 AudioQueue 给 TTS 播放。

    副作用放在 Node 层，因为：
    - 需要 access AudioQueue（通过 container 单例）
    - 需要 turn_id（从 state 里拿到）
    - PlannerBuilder 作为纯 Service，不应知道这些上下文
    """


    templates = TRANSITION_TEMPLATES.get(tool_name, ["好哦，我帮你搜一下"])
    transition_text = random.choice(templates)

    chunk = StreamChunk(
        text=transition_text,
        turn_id=turn_id,
        is_sentence_end=True,
        interruptible=True,
    )

    container.audio_queue.put(chunk)

    logger.info(
        f"[planner] 过渡回复已发送 | "
        f"turn_id={turn_id} | "
        f"text={transition_text}"
    )
