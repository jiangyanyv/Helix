from event.event import EventType


class RuntimeEventHandler:

    def __init__(

        self,

        runtime,

        audio_queue,

        tts_worker,

        event_bus

    ):

        self.runtime = runtime

        self.audio_queue = audio_queue

        self.tts = tts_worker

        self.bus = event_bus

        self.register()

    def register(self):
        self.bus.subscribe(

            EventType.USER_SPEAKING,

            self.on_user_start

        )

    def on_user_start(self, _):
        self.runtime.on_user_start()

        self.audio_queue.clear()

        self.tts.stop()