from core.conversation_graph import conversation_graph
from services.container import container
from services.llm.stream_chunk import StreamChunk
from loguru import logger


class Agent:
    """
    Agent入口

    负责:

    - session管理（通过 container.conversation_manager 单例，确保数据一致）
    - 调用 Conversation Graph（生成回复）
    - Turn 生命周期管理
    - 流式返回结果

    记忆抽取（Memory Graph）已改为跟随摘要触发：
    ConversationManager._check_summary_trigger 在摘要成功后，
    复用同一批待摘要消息异步执行 Memory Graph，
    不再在每轮对话末尾触发。


    不负责:

    - Prompt
    - Memory 查询/存储细节
    - LLM 调用细节

    """


    def __init__(self):

        self.conversation_graph = conversation_graph

        # 关键：必须使用 container 中的单例 ConversationManager，
        # 否则 Redis / 内存状态会与调用方视图不一致（类似之前 AudioQueue 多实例的坑）
        self.session_manager = container.conversation_manager

        self.runtime = container.runtime_manager


    def stream_chat(
            self,
            user_id: str,
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
            5. 保存 AI 消息（保存后由 ConversationManager 在摘要触发时一并触发记忆抽取）
            6. 结束 Turn
        """

        # =====================
        # 1. 保存用户消息
        # =====================

        self.session_manager.add_user_message(
            user_id,
            user_input
        )


        # =====================
        # 2. 获取历史消息
        # =====================

        messages = (
            self.session_manager
            .get_messages(user_id)
        )


        # =====================
        # 3. 启动 Turn
        # =====================

        turn = self.runtime.start_turn(user_id)
        turn_id = turn.turn_id
        # logger.info(f"Turn started | turn_id={turn_id}")


        # =====================
        # 4. 运行 Conversation Graph
        # =====================

        try:
            conv_result = self.conversation_graph.invoke(
                {
                    "user_id": user_id,
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
            user_id,
            response
        )


        # =====================
        # 6. 结束 Turn
        #
        # 记忆抽取已不在 Agent 主流程触发：
        # ConversationManager 在保存 AI 消息后，
        # 若达到摘要阈值会先做摘要，再复用同一批消息
        # 异步执行 Memory Graph（extractor → judge → updater）。
        # =====================

        self.runtime.finish_turn(turn_id)
        logger.info(f"Turn finished | turn_id={turn_id}")
