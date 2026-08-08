from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class StreamChunk:
    """
    LLM输出的一段可消费文本
    """

    # TTS真正需要转换的文本
    text: str

    # 当前Chunk属于哪个Turn
    turn_id: Optional[str] = None

    # 是否是完整句子
    is_sentence_end: bool = True

    # 当前Chunk对应的情绪
    emotion: str | None = None

    # 当前Chunk是否允许被打断
    interruptible: bool = True