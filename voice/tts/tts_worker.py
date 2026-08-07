import threading

from services.container import (
    container
)


class TTSWorker:

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