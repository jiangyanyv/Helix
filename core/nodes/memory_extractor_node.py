from __future__ import annotations

from core.state import AgentState
from services.container import container
from loguru import logger


def memory_extractor_node(state: AgentState):
    """
    Memory Graph Node

    调用 MemoryExtractor：
        用户消息 + AI 回复
            ↓
        LLM 结构化抽取
            ↓
        List[MemoryCandidate]
    """

    user_text = state.get("user_input", "") or ""
    ai_text = state.get("response", "") or ""

    candidates = container.memory_extractor.extract(
        user_text=user_text,
        ai_text=ai_text,
    )

    logger.info(
        f"[memory_extractor_node] "
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