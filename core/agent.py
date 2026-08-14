import threading

from core.conversation_graph import conversation_graph
from core.memory_graph import memory_graph
from services.container import container
from services.llm.stream_chunk import StreamChunk
from loguru import logger


class Agent:
    """
    Agent入口

    负责:

    - session管理（通过 container.conversation_manager 单例，确保数据一致）
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
            5. 保存 AI 消息
            6. 运行 Memory Graph：extractor → judge → updater（把本轮对话沉淀为长期记忆）
            7. 结束 Turn
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
        # 7. 异步运行 Memory Graph（写回记忆，不阻塞主流程）
        #
        # Memory Graph 内部会调用 LLM（MemoryExtractor / MemoryJudge），
        # 同步执行会让用户多等 1~3 秒。改为后台线程执行：
        #   - 主流程立即结束 Turn，用户可以马上开始下一轮输入
        #   - 后台线程完成记忆沉淀，失败不影响主流程
        #   - Container 中的 Service 单例均为线程安全：
        #       * Redis 客户端：连接池内置线程安全
        #       * MySQL scoped_session：线程本地 session，finally 中 remove()
        #       * LLM Client（OpenAI SDK）：线程安全
        # =====================

        """
        重构前暂时关闭
        """
        # self._run_memory_graph_async(
        #     user_id=user_id,
        #     user_input=user_input,
        #     response=response,
        # )


        # =====================
        # 8. 结束 Turn
        # =====================

        self.runtime.finish_turn(turn_id)
        logger.info(f"Turn finished | turn_id={turn_id}")


    # ==================================================
    # Memory Graph 异步执行
    # ==================================================

    def _run_memory_graph_async(
            self,
            user_id: str,
            user_input: str,
            response: str,
    ):
        """后台线程执行 Memory Graph，不阻塞主流程。

        - daemon=True：进程退出时自动结束，无需 join
        - 内部捕获所有异常，仅记录日志
        - 不返回结果（Memory Graph 是写回操作，无返回值需求）
        """

        thread = threading.Thread(
            target=self._memory_graph_task,
            args=(user_id, user_input, response),
            daemon=True,
            name=f"memory-graph-{user_id}",
        )

        thread.start()

        logger.info(
            f"Memory Graph 已调度后台执行 | "
            f"user_id={user_id} | "
            f"thread={thread.name}"
        )

    # ==================================================

    def _memory_graph_task(
            self,
            user_id: str,
            user_input: str,
            response: str,
    ):
        """Memory Graph 后台任务实体。

        在子线程中执行，所有异常都被捕获：
            - LangGraph 内部异常（Extractor/Judge/Updater）
            - DB 写入异常
            - LLM 调用异常
        都不会影响主流程。
        """

        try:

            self.memory_graph.invoke(
                {
                    "user_id": user_id,
                    "user_input": user_input,
                    "response": response,
                }
            )

            logger.info(
                f"Memory Graph 执行完成 | user_id={user_id}"
            )

        except Exception as e:  # noqa: BLE001

            logger.exception(
                f"Memory Graph 执行失败（不影响主流程） | "
                f"user_id={user_id} | error={e}"
            )
