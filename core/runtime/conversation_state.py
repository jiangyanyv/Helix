from enum import Enum


class ConversationState(str, Enum):

    IDLE = "idle"

    LISTENING = "listening"

    THINKING = "thinking"

    RESPONDING = "responding"