import threading

from services.container import (
    container
)


class TTSWorker:

    def __init__(self):
        # 新增停止标志
        self.stop_flag = False
        self.thread = None


    def start(self):

        thread = threading.Thread(

            target=self.run,

            daemon=True

        )

        thread.start()

    def run(self):

        while True:

            chunk = (
                container.audio_queue.get()
            )

            print(

                "[TTS]",

                chunk.text

            )

            # TODO

            # CosyVoice.generate(chunk)

    def interrupt(self):
        """中断TTS生成与播放"""
        self.stop_flag = True


    def reset(self):
        """重置停止标志，可重新开始新一轮TTS任务"""
        self.stop_flag = False