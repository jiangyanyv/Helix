from core.state import AgentState

from services.memory_pipeline.judge import MemoryJudge


judge = MemoryJudge()


def memory_judge_node(
        state: AgentState
):
    """
    Memory Graph Node

    输入:
        memory_candidates

    输出:
        accepted_memory
    """

    print(
        "====== Memory Judge ======"
    )


    accepted_memory = judge.judge(

        state.get(
            "memory_candidates",
            []
        )

    )


    return {

        "accepted_memory": accepted_memory

    }