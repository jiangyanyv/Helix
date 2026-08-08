from enum import Enum


class EventType(str, Enum):

    USER_SPEAKING = "user_speaking"

    USER_STOP = "user_stop"

    LLM_START = "llm_start"

    LLM_CHUNK = "llm_chunk"

    LLM_FINISH = "llm_finish"

    TTS_START = "tts_start"

    TTS_FINISH = "tts_finish"

    INTERRUPT = "interrupt"

    MEMORY_UPDATED = "memory_updated"