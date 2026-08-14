"""Memory Retriever LangGraph Node：
按本轮 user_input 查询 3 类新 Memory Service，组装 RetrievedMemory 写回 state。
不再走旧的 memory_manager.retrieve 分发接口（memory_manager.py 已删除）。
"""

from __future__ import annotations

from loguru import logger

from core.state import AgentState
from services.container import container
from services.memory.retrieved_memory import RetrievedMemory


def memory_retriever_node(state: AgentState):
    user_id = state.get("user_id", "")
    user_input = state.get("user_input", "") or ""

    profile: dict = {}
    relationships = []
    episodic = []

    # ============ 1) 用户画像（每次查 profile，小，LRU 已缓存） ============
    try:
        profile = container.profile_service.get(user_id) or {}
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[memory_retriever] profile 失败: {e}")
        profile = {}

    # ============ 2) 关系（仅召回当前消息中提到的人物） ============
    try:
        if user_input and user_input.strip():
            relationships = (
                    container.relationship_service.find_related(
                        user_id,
                        user_input,
                    )
                    or []
            )
        else:
            relationships = []
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[memory_retriever] relationships 失败: {e}")
        relationships = []

    # ============ 3) 相关事件（优先向量，降级最近N条） ============
    try:
        svc = container.episodic_service
        if user_input and user_input.strip():
            episodic = svc.search(user_id, user_input, top_k=5) or []
            for r in episodic:
                logger.debug(f"id={r['id']}, score={r['_score']:.4f}, content={r['content'][:50]}...")
        else:
            episodic = svc.search_recent(user_id, top_k=5) or []
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[memory_retriever] episodic 失败: {e}")
        episodic = []

    retrieved = RetrievedMemory(
        profile=profile,
        relationships=relationships,
        episodic=episodic,
    )

    logger.info(
        f"[memory_retriever] user_id={user_id} "
        f"profile_keys={list(profile.keys()) if profile else []}, "
        f"rels={len(relationships)}, epis={len(episodic)}"
    )
    return {"retrieved_memory": retrieved}
