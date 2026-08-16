from core.state import AgentState

from services.container import (
    container
)

from loguru import logger



def context_builder_node(
        state: AgentState
):


    retrieved_memory = state.get(
        "retrieved_memory"
    )

    tool_result = state.get(
        "tool_result"
    )

    if tool_result:
        logger.info(
            f"[context_builder] 收到 tool_result | "
            f"len={len(tool_result)}"
        )


    system_context = (
        container.context_builder
        .build(
            retrieved_memory=retrieved_memory,
            tool_result=tool_result,
        )
    )


    return {

        "system_context": system_context

    }