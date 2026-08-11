from core.state import AgentState
from services.container import container
from loguru import logger


def memory_updater_node(
        state: AgentState
):
    """
    Memory Graph Node

    调用Updater保存记忆（使用 container 中的 MemoryUpdater 单例，已注入3个Memory Service）

    """

    container.memory_updater.update(
        user_id=state["user_id"],
        memories=state.get("accepted_memory", [])
    )

    # logger.info("记忆写回完成")

    return {}