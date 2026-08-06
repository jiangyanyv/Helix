from typing import TypedDict
from typing import List

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    整个LangGraph共享状态
    """

    # 当前用户输入
    user_input: str

    # 对话历史
    messages: List[BaseMessage]

    # Perception
    intent: str
    emotion: str
    entities: List[str]

    # Memory
    memories: List[str]

    # Planner
    strategy: str

    # Prompt
    prompt: str

    # Response
    response: str