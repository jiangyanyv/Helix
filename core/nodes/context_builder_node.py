from core.state import AgentState

from services.container import (
    container
)



def context_builder_node(
        state: AgentState
):


    retrieved_memory = state.get(
        "retrieved_memory"
    )


    system_context = (
        container.context_builder
        .build(
            retrieved_memory
        )
    )


    return {

        "system_context": system_context

    }