from core.runtime.runtime_state import RuntimeState
from core.runtime.audio_state import AudioState
from core.runtime.conversation_state import ConversationState
from core.runtime.turn_manager import TurnManager


class RuntimeManager:

    """
    陪伴式 AI Runtime 管理器。

    RuntimeManager：
        管理整个系统运行状态。

    TurnManager：
        管理当前对话轮次生命周期。
    """

    def __init__(self):

        self.state = RuntimeState()

        self.turn_manager = TurnManager()

    # ==========================
    # 开始新的 Turn
    # ==========================

    def start_turn(
            self,
            session_id: str
    ):

        turn = self.turn_manager.start_turn(
            session_id
        )

        self.state.session_id = session_id

        self.state.current_turn_id = (
            turn.turn_id
        )

        self.state.interrupt_requested = False

        return turn

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
    # LLM开始
    # ==========================

    def on_llm_start(self):

        self.state.llm_generating = True

        self.state.conversation_state = (
            ConversationState.THINKING
        )

    # ==========================
    # LLM结束
    # ==========================

    def on_llm_finish(
            self,
            turn_id: str
    ):

        if not self.is_llm_accepting(turn_id):

            return False

        self.state.llm_generating = False

        return True

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

    def on_tts_finish(
            self,
            turn_id: str
    ):

        # 注意：这里不使用 is_llm_accepting，因为正常 finish 后 Turn.active=False
        # 但 TTS 仍在播放，播放完成后仍需要更新状态
        # 只要是当前 Turn 且未被打断，就允许更新状态
        if not self.is_tts_playable(turn_id):

            return False

        self.state.tts_playing = False

        self.state.audio_state = (
            AudioState.IDLE
        )

        self.state.conversation_state = (
            ConversationState.IDLE
        )

        return True

    # ==========================
    # 打断当前 Turn
    # ==========================

    def interrupt(
            self,
            turn_id: str | None = None
    ):

        interrupted = (
            self.turn_manager.interrupt_turn(
                turn_id
            )
        )

        if not interrupted:

            return False

        self.state.interrupt_requested = True

        self.state.llm_generating = False

        self.state.tts_playing = False

        self.state.audio_state = (
            AudioState.INTERRUPTED
        )

        return True

    # ==========================
    # 当前 Turn 是否接受 LLM 继续产出内容
    # （用于 LLM / Graph / Event 侧判断是否继续）
    #
    # - turn_id 匹配 且 active=True → 接受
    # - 其余（正常 finish / 被打断 / 已过期）→ 不接受
    # ==========================

    def is_llm_accepting(
            self,
            turn_id: str
    ) -> bool:

        return self.turn_manager.is_llm_accepting(
            turn_id
        )

    # ==========================
    # [辅助检查] 指定 Turn 是否已过期（被新 Turn 取代）
    #
    # 场景：用户开始了新一轮对话，旧 Turn 的残留 chunk 应当丢弃
    # ==========================

    def is_turn_expired(
            self,
            turn_id: str
    ) -> bool:

        return self.turn_manager.is_turn_expired(
            turn_id
        )

    # ==========================
    # [辅助检查] 指定 Turn 是否被用户打断
    #
    # 场景：用户中途插话打断，当前 Turn 的剩余未播放 chunk 应当丢弃
    # ==========================

    def is_turn_interrupted(
            self,
            turn_id: str
    ) -> bool:

        return self.turn_manager.is_turn_interrupted(
            turn_id
        )

    # ==========================
    # 判断指定 Turn 的 TTS chunk 是否允许播放
    #
    # - Turn 已过期（被新 Turn 取代） → 丢弃
    # - Turn 被用户打断             → 丢弃
    # - Turn 正常 finish（interrupted=False，即使 active=False）→ 允许
    # ==========================

    def is_tts_playable(
            self,
            turn_id: str
    ) -> bool:

        return self.turn_manager.is_tts_playable(
            turn_id
        )

    # ==========================
    # 完成 Turn
    # ==========================

    def finish_turn(
            self,
            turn_id: str
    ):

        finished = (
            self.turn_manager.finish_turn(
                turn_id
            )
        )

        if not finished:

            return False

        self.state.llm_generating = False

        self.state.tts_playing = False

        self.state.conversation_state = (
            ConversationState.IDLE
        )

        self.state.audio_state = (
            AudioState.IDLE
        )

        return True
