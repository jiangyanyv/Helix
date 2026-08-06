from typing import TypedDict, List
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    整个 Agent 的共享状态
    """

    # ========= 用户 =========
    user_input: str

    # ========= 对话 =========
    messages: List[BaseMessage]

    # ========= Perception =========
    intent: str
    emotion: str
    entities: List[str]

    # ========= Memory =========
    retrieved_memory: dict

    # ========= Planner =========
    strategy: str

    # ========= Prompt =========
    prompt: str

    # ========= Response =========
    response: str