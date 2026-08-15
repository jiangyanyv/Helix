from __future__ import annotations

from core.state import AgentState
from services.container import container
from loguru import logger


def memory_extractor_node(state: AgentState):
    """
    Memory Graph Node

    调用 MemoryExtractor：
        多轮对话消息
            ↓
        LLM 结构化批量抽取
            ↓
        List[MemoryCandidate]

    输入：state["messages"]（待抽取的多轮对话）
    输出：memory_candidates
    """

    messages = state.get("messages", []) or []

    candidates = container.memory_extractor.extract_from_messages(
        messages=messages,
    )

    logger.info(
        f"[memory_extractor_node] "
        f"msgs={len(messages)} "
        f"抽取候选记忆数量={len(candidates)}"
    )

    for c in candidates:
        logger.debug(
            f"[memory_extractor_node] "
            f"type={c.memory_type.value} "
            f"content={c.content!r} "
            f"importance={c.importance:.2f}"
        )

    return {
        "memory_candidates": candidates
    }
