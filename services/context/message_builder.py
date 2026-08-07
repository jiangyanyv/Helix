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


        messages = []


        if system_context:


            messages.append(
                SystemMessage(
                    content=system_context
                )
            )


        messages.extend(
            history
        )


        # messages.append(
        #     HumanMessage(
        #         content=user_input
        #     )
        # )


        return ChatRequest(

            model=model or self.DEFAULT_MODEL,

            messages=messages,

            stream=stream,

            temperature=temperature,

            top_p=top_p,

            max_tokens=max_tokens
        )