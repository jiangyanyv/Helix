from core.conversation_graph import conversation_graph
from core.memory_graph import memory_graph
from core.session.conversation_manager import ConversationManager
from services.container import container
from services.llm.stream_chunk import StreamChunk
from loguru import logger


class Agent:
    """
    Agent入口

    负责:

    - session管理
    - 调用 Conversation Graph（生成回复）
    - 调用 Memory Graph（写回记忆）
    - Turn 生命周期管理
    - 流式返回结果


    不负责:

    - Prompt
    - Memory 查询/存储细节
    - LLM 调用细节

    """


    def __init__(self):

        self.conversation_graph = conversation_graph

        self.memory_graph = memory_graph

        self.session_manager = ConversationManager()

        self.runtime = container.runtime_manager


    def stream_chat(
            self,
            session_id: str,
            user_input: str
    ):
        """
        流式对话。

        完整流程：
            1. 保存用户消息
            2. 启动新 Turn（产生 turn_id，TTS 依赖它判断有效性）
            3. 运行 Conversation Graph：
               memory_retriever → context_builder → message_builder → response_generator
               （response_generator 内部会把 StreamChunk 逐个放入 audio_queue 给 TTS）
            4. 流式 yield 回复文本给调用方
            5. 保存 AI 消息
            6. 运行 Memory Graph：extractor → judge → updater（把本轮对话沉淀为长期记忆）
            7. 结束 Turn
        """

        # =====================
        # 1. 保存用户消息
        # =====================

        self.session_manager.add_user_message(
            session_id,
            user_input
        )


        # =====================
        # 2. 获取历史消息
        # =====================

        messages = (
            self.session_manager
            .get_messages(session_id)
        )


        # =====================
        # 3. 启动 Turn
        # =====================

        turn = self.runtime.start_turn(session_id)
        turn_id = turn.turn_id
        logger.info(f"Turn started | turn_id={turn_id}")


        # =====================
        # 4. 运行 Conversation Graph
        # =====================

        try:
            conv_result = self.conversation_graph.invoke(
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "user_input": user_input,
                    "messages": messages
                }
            )
        except Exception as e:
            logger.exception(f"Conversation Graph 执行失败: {e}")
            self.runtime.finish_turn(turn_id)
            raise


        # =====================
        # 5. 流式 yield 回复文本
        # =====================

        response_chunks: list[StreamChunk] = conv_result.get(
            "response_chunks",
            []
        )

        response = conv_result.get(
            "response",
            ""
        )

        for chunk in response_chunks:
            yield chunk.text


        # =====================
        # 6. 保存 AI 回复
        # =====================

        self.session_manager.add_ai_message(
            session_id,
            response
        )


        # =====================
        # 7. 运行 Memory Graph（写回记忆，失败不影响主流程）
        # =====================

        try:
            self.memory_graph.invoke(
                {
                    "session_id": session_id,
                    "user_input": user_input,
                    "response": response,
                }
            )
            # logger.debug("Memory Graph 执行完成")
        except Exception as e:
            logger.exception(f"Memory Graph 执行失败（不影响主流程）: {e}")


        # =====================
        # 8. 结束 Turn
        # =====================

        self.runtime.finish_turn(turn_id)
        logger.info(f"Turn finished | turn_id={turn_id}")
