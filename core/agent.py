"""Agent 入口。

负责:
- session管理（通过 container.conversation_manager 单例）
- 调用 Conversation Graph（生成回复）
- Turn 生命周期管理
- 流式返回结果

Conversation Graph 内部根据 Config.ENABLE_TOOL_CALLING 自动选择：
    状态A：memory_retriever → context_builder → message_builder → response_generator
    状态B：planner → conditional_edge
              ├─ normal → memory_retriever → ...
              └─ tool   → task_agent(过渡回复+工具) → context_builder → ...

记忆抽取（Memory Graph）跟随摘要触发，不在主流程。
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from core.conversation_graph import conversation_graph
from services.container import container
from services.llm.stream_chunk import StreamChunk


class Agent:
    """Agent 入口。"""

    def __init__(self):

        self.conversation_graph = conversation_graph

        # 关键：必须使用 container 中的单例 ConversationManager
        self.session_manager = container.conversation_manager

        self.runtime = container.runtime_manager

    def stream_chat(
        self,
        user_id: str,
        user_input: str,
    ):
        """流式对话。

        完整流程：
            1. 保存用户消息
            2. 获取历史消息
            3. 启动 Turn（产生 turn_id）
            4. 运行 Conversation Graph（graph 内部根据状态自动路由）
               - 状态B 且 need_tool=true 时：
                 planner 先放过渡回复到 AudioQueue（TTS 立即播放）
                 → task_agent 执行工具 → context_builder 注入 tool_result
                 → response_generator 生成最终回复
            5. 流式 yield 回复文本给调用方
            6. 保存 AI 消息（含 tool_result 元数据 if 状态B）
            7. 结束 Turn
        """

        # =====================
        # 1. 保存用户消息
        # =====================

        self.session_manager.add_user_message(
            user_id,
            user_input,
        )

        # =====================
        # 2. 获取历史消息
        # =====================

        messages = self.session_manager.get_messages(user_id)

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
                    "messages": messages,
                }
            )
        except Exception as e:
            logger.exception(
                f"Conversation Graph 执行失败: {e}"
            )
            self.runtime.finish_turn(turn_id)
            raise

        # =====================
        # 5. 流式 yield 回复文本
        # =====================

        response_chunks = conv_result.get(
            "response_chunks", []
        )
        response = conv_result.get("response", "")

        for chunk in response_chunks:
            yield chunk.text

        # =====================
        # 6. 保存 AI 消息
        #
        # 状态B 且使用了工具时，把 tool_result 存入
        # additional_kwargs 供程序逻辑读取。
        # LLM 下一轮通过 content 自然感知工具结果。
        # =====================

        tool_result = conv_result.get("tool_result")
        tool_name = conv_result.get("tool_name")

        ai_kwargs = {}

        if tool_result:
            ai_kwargs["tool_result"] = tool_result
            ai_kwargs["used_tool"] = tool_name

        self.session_manager.add_ai_message(
            user_id,
            response,
            additional_kwargs=ai_kwargs or None,
        )

        # =====================
        # 7. 结束 Turn
        # =====================

        self.runtime.finish_turn(turn_id)
        logger.info(f"Turn finished | turn_id={turn_id}")
