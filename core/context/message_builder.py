from services.llm.chat_request import ChatRequest
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)
from config import Config

class MessageBuilder:
    """
    构建发送给LLM的ChatRequest

    这里只负责组装数据
    不负责任何业务逻辑
    """

    DEFAULT_MODEL = Config.LLM_MODEL

    def build(
            self,
            *,
            system_context: str,
            history: list,
            # user_input: str,
            model: str | None = None,
            stream: bool = True,
            temperature: float = 0.9,
            top_p: float = 0.95,
            max_tokens=None

    ) -> ChatRequest:


        # messages = []
        #
        #
        # if system_context:
        #
        #
        #     messages.append(
        #         SystemMessage(
        #             content=system_context
        #         )
        #     )
        #
        #
        # messages.extend(history)

        system_parts = []

        # ========================================================
        # 1. 当前 Agent 的 System Context
        # ========================================================

        if system_context:
            system_parts.append(system_context)

        # ========================================================
        # 2. 从 history 中提取 SystemMessage
        #
        # 目前主要就是 ConversationManager 生成的：
        #
        # SystemMessage(
        #     "【历史对话摘要】..."
        # )
        # ========================================================

        conversation_messages = []

        for message in history:

            if isinstance(message, SystemMessage):
                if message.content:
                    system_parts.append(
                        str(message.content)
                    )

            else:
                conversation_messages.append(
                    message
                )

        # ========================================================
        # 3. 构造最终唯一的 SystemMessage
        # ========================================================

        messages = []

        if system_parts:
            messages.append(
                SystemMessage(
                    content="\n\n".join(system_parts)
                )
            )

        # ========================================================
        # 4. 保留 HumanMessage / AIMessage
        # ========================================================

        messages.extend(
            conversation_messages
        )



        return ChatRequest(

            model=model or self.DEFAULT_MODEL,

            messages=messages,

            stream=stream,

            temperature=temperature,

            top_p=top_p,

            max_tokens=max_tokens
        )