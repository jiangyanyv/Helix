from core.state import AgentState
from services.container import container


def memory_judge_node(
        state: AgentState
):
    """
    Memory Graph Node

    输入:
        memory_candidates

    输出:
        accepted_memory

    使用 container 中的 MemoryJudge 单例
    """

    accepted_memory = container.memory_judge.judge(
        state.get("memory_candidates", [])
    )

    return {
        "accepted_memory": accepted_memory
    }