import time

from openai import OpenAI

from services.llm.client import LLMClient
from services.llm.chat_request import ChatRequest
from services.llm.chat_response import ChatResponse
from config import Config
from loguru import logger

class DeepSeekClient(
    LLMClient
):

    def __init__(
            self,
            api_key: str=Config.OPENAI_API_KEY,
            base_url: str=Config.OPENAI_BASE_URL,
            model: str=Config.LLM_MODEL,
    ):

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model = model


    def generate(
            self,
            request: ChatRequest
    ) -> ChatResponse:

        kwargs = self._build_kwargs(request)

        response = self.client.chat.completions.create(**kwargs)

        content = (
            response
            .choices[0]
            .message
            .content
        )

        usage = getattr(response, "usage", None)

        return ChatResponse(
            text=content or "",
            finish_reason=response.choices[0].finish_reason or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )


    def stream(
            self,
            request: ChatRequest
    ):

        kwargs = self._build_kwargs(request)

        kwargs["stream"] = True

        logger.info(request)
        '''llm端到端时间记录代码1'''
        # logger.info("开始调用llm...")
        # first_token_time = None
        # token_count = 0
        # start = time.time()


        response = self.client.chat.completions.create(
            **kwargs
        )

        for chunk in response:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if not delta:
                continue

            content = delta.content

            if not content:
                continue


            '''llm端到端时间记录代码2'''
            # if first_token_time is None:
            #     first_token_time = time.time()
            #     logger.info(
            #         f"首Token耗时(TTFT): "
            #         f"{first_token_time - start:.3f}s"
            #     )
            # token_count += 1


            # 这里只负责Token
            yield content

        '''llm端到端时间记录代码3'''
        # total = time.time() - start
        # logger.info(
        #     f"总耗时:{total:.3f}s "
        #     f"| Token:{token_count}"
        # )


    ROLE_MAP = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }

    def _convert_messages(self, chat_request: ChatRequest) -> list:
        """将 ChatRequest 的 messages 转换为 OpenAI 格式的 dict 列表"""
        result = []
        for msg in chat_request.messages:
            """ 
            langchain BaseMessage -> dict
            type: system / human / ai  -> role: system / user / assistant
            """
            role = self.ROLE_MAP.get(msg.type, msg.type)
            result.append({
                "role": role,
                "content": msg.content
            })
        return result


    def _build_kwargs(self, request: ChatRequest) -> dict:
        """构建 OpenAI API 请求参数，自动过滤无效值"""
        kwargs = {
            "model": request.model or self.model,
            "messages": self._convert_messages(request),
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        # 仅在 max_tokens 为有效正整数时传入
        if request.max_tokens and request.max_tokens > 0:
            kwargs["max_tokens"] = request.max_tokens
        return kwargs


