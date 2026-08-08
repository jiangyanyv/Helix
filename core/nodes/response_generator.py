from core.state import AgentState

from services.container import container
from services.event.event import EventType



def response_generator_node(
        state: AgentState
):


    chat_request = state.get(
        "chat_request"
    )

    turn_id = state.get("turn_id")


    if not chat_request:

        return {

            "response_chunks": []

        }

    chunks = []

    texts = []

    # 发布 LLM 开始事件
    container.event_bus.publish(
        EventType.LLM_START,
        {"turn_id": turn_id}
    )

    try:
        for chunk in container.response_service.stream(
            chat_request,
            turn_id=turn_id,
        ):
            container.event_bus.publish(

                EventType.LLM_CHUNK,

                chunk

            )

            chunks.append(chunk)

            texts.append(chunk.text)

            container.audio_queue.put(
                chunk
            )
    finally:
        # 发布 LLM 结束事件
        container.event_bus.publish(
            EventType.LLM_FINISH,
            {"turn_id": turn_id}
        )

    return {

        "response_chunks": chunks,

        "response": "".join(texts)

    }