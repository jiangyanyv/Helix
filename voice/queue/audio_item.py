from dataclasses import dataclass

from services.llm.stream_chunk import (
    StreamChunk
)


@dataclass(slots=True)
class AudioItem:

    chunk: StreamChunk

    priority: int = 0

    timestamp: float = 0