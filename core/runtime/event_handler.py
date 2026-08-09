from services.event.event import EventType


class RuntimeEventHandler:

    def __init__(
            self,
            runtime_manager,
            audio_queue,
            tts_worker,
            event_bus
    ):

        self.runtime = runtime_manager

        self.audio_queue = audio_queue

        self.tts = tts_worker

        self.event_bus = event_bus

        self.register()

    def register(self):

        self.event_bus.subscribe(
            EventType.USER_SPEAKING,
            self.on_user_speaking
        )

        self.event_bus.subscribe(
            EventType.TTS_START,
            self.on_tts_start
        )

        self.event_bus.subscribe(
            EventType.TTS_FINISH,
            self.on_tts_finish
        )

        self.event_bus.subscribe(
            EventType.INTERRUPT,
            self.on_interrupt
        )

    # ==========================
    # 用户开始说话
    # ==========================

    def on_user_speaking(self, turn_id):

        print(
            f"🎤 用户开始说话 | turn={turn_id}"
        )

        self.runtime.on_user_start()

    # ==========================
    # TTS开始
    # ==========================

    def on_tts_start(self, turn_id):

        # TTS_START 事件要生效，需要 Turn 对 TTS 仍然有效
        # （被打断 / 已过期的 Turn 不应该再更新 tts_playing 状态）
        if not self.runtime.is_tts_playable(
            turn_id
        ):

            return

        self.runtime.on_tts_start()

    # ==========================
    # TTS结束
    # ==========================

    def on_tts_finish(self, turn_id):

        self.runtime.on_tts_finish(
            turn_id
        )

    # ==========================
    # 打断
    # ==========================

    def on_interrupt(self, turn_id):

        print(
            f"INTERRUPT | turn={turn_id}"
        )

        success = self.runtime.interrupt(
            turn_id
        )

        if not success:
            return

        # 第一层：清理尚未消费的 Chunk
        self.audio_queue.clear()

        # 第二层：停止当前正在播放的 TTS
        self.tts.stop()