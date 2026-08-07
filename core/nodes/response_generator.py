from core.state import AgentState

from services.container import container
from services.event.event import EventType



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

    texts = []

    for chunk in container.response_service.stream(chat_request):
        container.event_bus.publish(

            EventType.LLM_CHUNK,

            chunk

        )

        chunks.append(chunk)

        texts.append(chunk.text)

        container.audio_queue.put(
            chunk
        )

    return {

        "response_chunks": chunks,

        "response": "".join(texts)

    }