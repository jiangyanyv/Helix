from core.state import AgentState

from services.memory_pipeline.updater import MemoryUpdater

from loguru import logger

updater = MemoryUpdater()



def memory_updater_node(
        state: AgentState
):
    """
    Memory Graph Node

    调用Updater保存记忆

    """




    updater.update(

        user_id=state["user_id"],

        memories=state.get(
            "accepted_memory",
            []
        )

    )

    logger.info("Fake记忆存储成功")

    return {}