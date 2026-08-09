from dataclasses import dataclass
from uuid import uuid4


@dataclass(slots=True)
class Turn:
    """
    一轮用户 -> AI 对话的上下文标识。
    """

    turn_id: str

    user_id: str

    # LLM 生成阶段是否仍在进行（finish/interrupt 后变为 False）
    active: bool = True

    # 是否被用户打断（打断后剩余 TTS chunk 应丢弃；正常 finish 的 interrupted 仍为 False）
    interrupted: bool = False


class TurnManager:
    """
    管理当前 Session 的对话 Turn。

    一个 Turn 对应一次用户输入以及由此产生的 AI 回复。

    例如：

        Turn A
        用户：你好
        AI：你好呀……

        用户打断

        Turn A -> inactive + interrupted

        Turn B
        用户：等等，我想问个问题
    """

    def __init__(self):

        self.current_turn: Turn | None = None

    # ==========================
    # 创建新的 Turn
    # ==========================

    def start_turn(
            self,
            user_id: str
    ) -> Turn:

        # 如果之前存在 Turn
        # 先让旧 Turn 失效，并标记为被打断（丢弃剩余 TTS chunk）
        if self.current_turn is not None:

            self.current_turn.active = False

            self.current_turn.interrupted = True

        turn = Turn(

            turn_id=uuid4().hex,

            user_id=user_id,

            active=True,

            interrupted=False

        )

        self.current_turn = turn

        return turn

    # ==========================
    # 当前 Turn 是否接受 LLM 继续产出内容
    # （用于 LLM / Graph 侧判断是否继续）
    #
    # 等价语义：是否处于「LLM 活跃阶段」
    # - turn_id 匹配 + active=True  → 接受
    # - 其余（正常 finish / 被打断 / 已过期）→ 不接受
    # ==========================

    def is_llm_accepting(
            self,
            turn_id: str
    ) -> bool:

        if self.current_turn is None:

            return False

        return (

            self.current_turn.turn_id == turn_id

            and

            self.current_turn.active

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

        if self.current_turn is None:

            return True

        return self.current_turn.turn_id != turn_id

    # ==========================
    # [辅助检查] 指定 Turn 是否被用户打断
    #
    # 场景：用户中途插话打断，当前 Turn 的剩余未播放 chunk 应当丢弃
    # 注意：该检查前提是 turn_id 与 current_turn 匹配（未过期）
    # ==========================

    def is_turn_interrupted(
            self,
            turn_id: str
    ) -> bool:

        if self.current_turn is None:

            return False

        if self.current_turn.turn_id != turn_id:

            return False

        return self.current_turn.interrupted

    # ==========================
    # 判断指定 Turn 的 TTS chunk 是否允许播放
    #
    # - Turn 已过期（被新 Turn 取代） → 不允许，丢弃
    # - Turn 被用户打断           → 不允许，丢弃
    # - Turn 正常 finish（interrupted=False，即使 active=False）→ 允许
    # ==========================

    def is_tts_playable(
            self,
            turn_id: str
    ) -> bool:

        if self.is_turn_expired(turn_id):

            return False

        if self.is_turn_interrupted(turn_id):

            return False

        # turn_id 匹配 且 未被打断 → 允许（不管 active 是否为 False）
        return True

    # ==========================
    # 正常结束当前 Turn
    # （只置 active=False，保持 interrupted=False，TTS 可以继续播放完）
    # ==========================

    def finish_turn(
            self,
            turn_id: str
    ) -> bool:

        if not self.is_llm_accepting(turn_id):

            return False

        self.current_turn.active = False

        return True

    # ==========================
    # 打断当前 Turn
    # （置 active=False 且 interrupted=True，剩余 TTS chunk 会被丢弃）
    # ==========================

    def interrupt_turn(
            self,
            turn_id: str | None = None
    ) -> bool:

        if self.current_turn is None:

            return False

        # 如果指定了 turn_id，
        # 必须是当前 Turn 才允许打断
        if (
                turn_id is not None
                and self.current_turn.turn_id != turn_id
        ):

            return False

        self.current_turn.active = False

        self.current_turn.interrupted = True

        return True

    # ==========================
    # 获取当前 Turn
    # ==========================

    def get_current_turn(self) -> Turn | None:

        return self.current_turn