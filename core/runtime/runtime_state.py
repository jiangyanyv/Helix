from dataclasses import dataclass

from core.runtime.audio_state import AudioState
from core.runtime.conversation_state import ConversationState


@dataclass(slots=True)
class RuntimeState:

    user_id: str = ""

    current_turn_id: str = ""

    current_response_id: str = ""

    conversation_state: ConversationState = (
        ConversationState.IDLE
    )

    audio_state: AudioState = (
        AudioState.IDLE
    )

    llm_generating: bool = False

    tts_playing: bool = False

    user_speaking: bool = False

    interrupt_requested: bool = False