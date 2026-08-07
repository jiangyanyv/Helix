from dataclasses import dataclass, field
from typing import List, Dict, Any

from langchain_core.messages import BaseMessage


@dataclass(slots=True)
class ChatRequest:
    """
    LLM 请求对象

    Agent 与 LLMClient 之间统一的数据结构
    """

    # OpenAI messages
    messages: list[BaseMessage]

    # 模型名称
    model: str

    # 是否流式
    stream: bool = True

    # 推理参数
    temperature: float = 0.8

    top_p: float = 0.95

    max_tokens: int | None = None

    # 预留
    extra: Dict[str, Any] = field(default_factory=dict)