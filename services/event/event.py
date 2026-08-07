from enum import Enum


class EventType(str, Enum):

    LLM_CHUNK = "llm_chunk"

    TTS_START = "tts_start"

    TTS_FINISH = "tts_finish"

    MEMORY_UPDATED = "memory_updated"

    USER_INTERRUPT = "user_interrupt"

    ASR_RESULT = "asr_result"