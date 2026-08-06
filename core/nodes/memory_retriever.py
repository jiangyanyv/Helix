from core.state import AgentState

from services.context.context_builder import ContextBuilder


context_builder = ContextBuilder()


def memory_retriever_node(
        state: AgentState
):
    """
    LangGraph Node

    负责：
    调用ContextBuilder

    不负责：
    Memory检索逻辑
    """

    print(
        "====== Memory Retriever ======"
    )


    retrieved_memory = context_builder.build(

        session_id=state["session_id"],

        query=state["user_input"]

    )


    return {

        "retrieved_memory": retrieved_memory

    }