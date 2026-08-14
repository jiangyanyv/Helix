import threading

from loguru import logger

from services.llm.stream_chunk import StreamChunk


class TTSWorker:

    def __init__(
            self,
            audio_queue,
            runtime_manager
    ):

        self.audio_queue = audio_queue

        self.runtime = runtime_manager

        self._stop_event = threading.Event()

        self._thread = None

        # 测试 / 调试用
        self.played_chunks = []

    # ==================================================
    # Start
    # ==================================================

    def start(self):

        self._thread = threading.Thread(
            target=self.run,
            daemon=True
        )

        self._thread.start()

    # ==================================================
    # Main Loop
    # ==================================================

    def run(self):

        logger.info("TTS Worker started")
        # print("🔊 TTS Worker started")

        while True:

            chunk = self.audio_queue.get()

            if chunk is None:
                continue

            # ==================================================
            # 第一层检查
            # 当前 Turn 的 TTS 是否允许播放
            #
            # - 正常 finish 的 Turn（interrupted=False）：
            #   继续播放，不丢弃（即使 active=False）
            # - 被用户打断（interrupted=True）：
            #   丢弃当前 Turn 的剩余未播放 chunk
            # - Turn 已过期（被新 Turn 取代，turn_id 不匹配）：
            #   丢弃旧 Turn 的残留 chunk
            # ==================================================

            if self.runtime.is_turn_expired(chunk.turn_id):

                # 场景：用户又输入了新内容，开始了新一轮对话
                #       旧 Turn 还没播完的 chunk 全部作废
                print(
                    f"🛑 丢弃（Turn已过期，新一轮对话已开始） | "
                    f"turn={chunk.turn_id} | "
                    f"text={chunk.text!r}"
                )

                continue

            if self.runtime.is_turn_interrupted(chunk.turn_id):

                # 场景：用户中途插话打断了当前 Turn
                #       正在播放的 chunk 已由 tts.stop() 处理
                #       这里丢弃的是「同一 Turn 中，正在播放 chunk 之后的后续内容」
                print(
                    f"🛑 丢弃（被用户打断） | "
                    f"turn={chunk.turn_id} | "
                    f"text={chunk.text!r}"
                )

                continue

            # ==================================================
            # 第二层
            # 开始播放
            # ==================================================

            '''测试功能暂停'''
            # print(
            #     f"🔊 TTS播放 | "
            #     f"turn={chunk.turn_id} | "
            #     f"text={chunk.text!r}"
            # )

            # 测试已播放
            self.played_chunks.append(chunk)

            # 告诉 Runtime：
            # 当前正在播放 TTS
            self.runtime.on_tts_start()

            try:

                self._play(chunk)

            finally:

                self.runtime.on_tts_finish(
                    chunk.turn_id
                )

    # ==================================================
    # 模拟 TTS 播放
    # ==================================================

    def _play(
            self,
            chunk: StreamChunk
    ):

        # ==================================================
        # interruptible = False
        #
        # 当前 Chunk 不允许被打断
        # ==================================================

        if not chunk.interruptible:

            print(
                "🔒 当前 Chunk 不允许打断"
            )

            threading.Event().wait(
                timeout=1.0
            )

            return

        # ==================================================
        # interruptible = True
        #
        # 当前 Chunk 可以被打断
        # ==================================================

        interrupted = self._stop_event.wait(
            timeout=1.0
        )

        if interrupted:

            print(
                "⛔ 当前 TTS 播放被打断"
            )

            self._stop_event.clear()

    # ==================================================
    # Stop
    # ==================================================

    def stop(self):

        print(
            "⛔ TTS stop"
        )

        self._stop_event.set()

    # ==================================================
    # Status
    # ==================================================

    def is_running(self):

        return self._thread is not None