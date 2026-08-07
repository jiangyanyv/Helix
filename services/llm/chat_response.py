from dataclasses import dataclass


@dataclass(slots=True)
class ChatResponse:
    """
    LLM 返回结果
    """

    text: str

    finish_reason: str = ""

    prompt_tokens: int = 0

    completion_tokens: int = 0

    total_tokens: int = 0

    latency_ms: float = 0.0