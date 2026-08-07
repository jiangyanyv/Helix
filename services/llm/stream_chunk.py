from dataclasses import dataclass


@dataclass(slots=True)
class StreamChunk:
    """
    LLM输出的一段可消费文本
    """

    text: str

    is_sentence_end: bool = True

    emotion: str | None = None

    interruptible: bool = True