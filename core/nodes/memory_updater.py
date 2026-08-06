from core.state import AgentState

from services.memory_pipeline.updater import MemoryUpdater



updater = MemoryUpdater()



def memory_updater_node(
        state: AgentState
):
    """
    Memory Graph Node

    调用Updater保存记忆

    """


    print(
        "====== Memory Updater ======"
    )


    updater.update(

        session_id=state["session_id"],

        memories=state.get(
            "accepted_memory",
            []
        )

    )


    return {}