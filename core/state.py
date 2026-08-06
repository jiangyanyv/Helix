from typing import TypedDict, List


class AgentState(TypedDict):
    """
    整个LangGraph共享状态
    """

    # ========= 用户输入 =========
    user_input: str

    # ========= Perception =========
    intent: str
    emotion: str
    entities: List[str]

    # ========= Memory =========
    memories: List[str]

    # ========= Planner =========
    strategy: str

    # ========= LLM =========
    response: str