from enum import Enum


class AudioState(str, Enum):

    IDLE = "idle"

    PLAYING = "playing"

    INTERRUPTED = "interrupted"