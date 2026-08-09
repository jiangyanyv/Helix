from core.state import AgentState

from services.memory_pipeline.extractor import MemoryExtractor


extractor = MemoryExtractor()


def memory_extractor_node(
        state: AgentState
):
    """
    Memory Graph Node

    职责：

    调用Extractor Service

    """



    candidates = extractor.extract(

        user_text=state["user_input"],

        ai_text=state.get(
            "response",
            ""
        )

    )


    return {

        "memory_candidates": candidates

    }