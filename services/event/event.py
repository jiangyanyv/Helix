from enum import Enum


class EventType(str, Enum):

    LLM_CHUNK = "llm_chunk"

    TTS_START = "tts_start"

    TTS_FINISH = "tts_finish"

    MEMORY_UPDATED = "memory_updated"

    USER_INTERRUPT = "user_interrupt"

    ASR_RESULT = "asr_result"

    ASR_START = "asr_start"

    ASR_FINISH = "asr_finish"

    USER_SPEAKING = "user_speaking"

    USER_STOP = "user_stop"

    LLM_START = "llm_start"

    LLM_FINISH = "llm_finish"


    INTERRUPT = "runtime"
