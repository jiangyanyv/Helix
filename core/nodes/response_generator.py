from core.state import AgentState

from services.container import container



def response_generator_node(
        state: AgentState
):


    chat_request = state.get(
        "chat_request"
    )


    if not chat_request:

        return {

            "response_chunks": []

        }



    chunks = []


    for chunk in (
        container.response_service
        .stream(chat_request)
    ):

        chunks.append(chunk)


    return {

        "response_chunks": chunks,

        "response": "".join(chunks)

    }