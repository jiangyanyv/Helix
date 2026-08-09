from core.state import AgentState

from services.container import (
    container
)



def memory_retriever_node(
        state: AgentState
):


    session_id = state.get(
        "session_id",
        ""
    )


    user_input = state.get(
        "user_input",
        ""
    )


    retrieved_memory = (
        container.memory_manager
        .retrieve(

            session_id,

            user_input

        )
    )


    return {


        "retrieved_memory":

        retrieved_memory


    }