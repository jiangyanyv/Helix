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

from core.state import AgentState
from services.container import container
from services.llm.stream_chunk import StreamChunk


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

    try:
        # Step1: 构建请求（Service 层纯逻辑）
        request = container.planner_builder.build(user_input)

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
