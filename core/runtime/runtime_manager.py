from core.runtime.runtime_state import (
    RuntimeState
)

from core.runtime.audio_state import (
    AudioState
)

from core.runtime.conversation_state import (
    ConversationState
)


class RuntimeManager:

    """
    AI AGENT 运行时管理器

    全局唯一
    """

    def __init__(self):

        self.state = RuntimeState()

    # ==========================
    # 用户开始说话
    # ==========================

    def on_user_start(self):

        self.state.user_speaking = True

        self.state.conversation_state = (
            ConversationState.LISTENING
        )

    # ==========================
    # 用户停止说话
    # ==========================

    def on_user_stop(self):

        self.state.user_speaking = False

    # ==========================
    # LLM开始生成
    # ==========================

    def on_llm_start(self):

        self.state.llm_generating = True

        self.state.conversation_state = (
            ConversationState.THINKING
        )

    # ==========================
    # LLM结束
    # ==========================

    def on_llm_finish(self):

        self.state.llm_generating = False

        self.state.conversation_state = (
            ConversationState.IDLE
        )

    # ==========================
    # TTS开始
    # ==========================

    def on_tts_start(self):

        self.state.tts_playing = True

        self.state.audio_state = (
            AudioState.PLAYING
        )

    # ==========================
    # TTS结束
    # ==========================

    def on_tts_finish(self):

        self.state.tts_playing = False

        self.state.audio_state = (
            AudioState.IDLE
        )