from core.state import AgentState

from services.container import (
    container
)



def message_builder_node(
        state: AgentState
):


    chat_request = (
        container.message_builder
        .build(

            system_context=
                state.get(
                    "system_context",
                    ""
                ),

            history=
                state.get(
                    "messages",
                    []
                ),

            # user_input=
            #     state.get(
            #         "user_input",
            #         ""
            #     )
        )
    )


    return {

        "chat_request": chat_request

    }