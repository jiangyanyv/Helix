from loguru import logger

from core.state import AgentState
from services.container import container
from services.event.event import EventType
from services.response.fallback import create_fallback_chunk


def response_generator_node(
    state: AgentState,
):
    """
    Response Generator Node

    职责：
    1. 调用 ResponseService 获取 LLM 流式响应
    2. 发布 LLM_START / LLM_CHUNK / LLM_FINISH 事件
    3. 将响应 chunk 放入 AudioQueue
    4. LLM 失败时提供人设化兜底回复
    """

    chat_request = state.get("chat_request")
    turn_id = state.get("turn_id")

    # 没有请求时，不进行 LLM 调用
    if not chat_request:
        return {
            "response_chunks": [],
            "response": "",
        }

    chunks = []
    texts = []

    # ============================================================
    # LLM 开始
    # ============================================================

    container.event_bus.publish(
        EventType.LLM_START,
        {"turn_id": turn_id},
    )

    try:
        # ========================================================
        # LLM 流式调用
        # ========================================================

        for chunk in container.response_service.stream(
            chat_request,
            turn_id=turn_id,
        ):
            chunks.append(chunk)
            texts.append(chunk.text)

            # 发布 LLM Chunk 事件
            container.event_bus.publish(
                EventType.LLM_CHUNK,
                chunk,
            )

            # 进入 TTS / AudioQueue
            container.audio_queue.put(chunk)

    except Exception:
        # ========================================================
        # LLM / Response Pipeline 异常
        #
        # 不向用户暴露具体异常。
        # 使用人设化 fallback 保证本轮对话仍然能够结束。
        # ========================================================

        logger.exception(
            "[response_generator] response generation failed "
            f"| turn_id={turn_id}"
        )

        # --------------------------------------------------------
        # 当前流式结果可能是不完整的。
        #
        # 如果已经产生了部分内容：
        #
        #   "主人今天有没有……"
        #
        # 直接继续播放可能造成语义断裂。
        #
        # 当前阶段先采用：
        #   丢弃当前未完成结果 → 使用完整 fallback
        #
        # 后续可以根据 sentence boundary 做更精细的处理。
        # --------------------------------------------------------

        chunks.clear()
        texts.clear()

        # 清理尚未被 TTS 消费的 chunk
        container.audio_queue.clear()

        # --------------------------------------------------------
        # 创建 fallback
        # --------------------------------------------------------

        fallback_chunk = create_fallback_chunk(
            turn_id=turn_id,
        )

        chunks.append(fallback_chunk)
        texts.append(fallback_chunk.text)

        # 让 fallback 正常进入 TTS
        container.audio_queue.put(
            fallback_chunk
        )

        # 发布 fallback 对应的 LLM_CHUNK
        container.event_bus.publish(
            EventType.LLM_CHUNK,
            fallback_chunk,
        )

    finally:
        # ========================================================
        # 无论 LLM 成功还是失败，都必须结束 LLM 阶段
        #
        # 防止 Runtime 状态一直停留在 THINKING。
        # ========================================================

        container.event_bus.publish(
            EventType.LLM_FINISH,
            {"turn_id": turn_id},
        )

    return {
        "response_chunks": chunks,
        "response": "".join(texts),
    }