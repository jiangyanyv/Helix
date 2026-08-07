from queue import Queue

from services.llm.stream_chunk import (
    StreamChunk
)


class AudioQueue:
    """
    LLM -> TTS

    文本队列
    """

    def __init__(self):

        self.queue = Queue()

    def put(
            self,
            chunk: StreamChunk
    ):

        self.queue.put(chunk)

    def get(self):

        return self.queue.get()

    def empty(self):

        return self.queue.empty()