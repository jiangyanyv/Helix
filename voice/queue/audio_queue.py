from queue import Queue, Empty
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

        self._stop_flag = False

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
        """
        清空当前所有待播放文本。
        """

        while True:

            try:

                self.queue.get_nowait()

            except Empty:

                break

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