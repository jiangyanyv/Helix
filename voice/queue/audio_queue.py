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

    def clear(self):
        """清空队列所有缓存数据"""
        while not self.queue.empty():
            self.queue.get_nowait()

    def stop(self):
        """标记队列停止，同时清空已有数据"""
        self._stop_flag = True
        self.clear()

    def is_stopped(self):
        """对外查询是否已触发停止"""
        return self._stop_flag

    def reset(self):
        """重置停止标记，恢复队列正常使用"""
        self._stop_flag = False