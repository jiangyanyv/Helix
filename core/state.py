from typing import TypedDict, List

from langchain_core.messages import BaseMessage

from services.llm.chat_request import ChatRequest
from services.memory.retrieved_memory import RetrievedMemory
from services.llm.stream_chunk import StreamChunk

from services.memory.memory_candidate import MemoryCandidate



class AgentState(TypedDict, total=False):
    """
    LangGraph 全局状态

    所有Node共享

    """

    # =====================
    # Session & Turn
    # =====================

    user_id: str

    # 当前对话轮次ID（由 RuntimeManager.start_turn 创建）
    turn_id: str

    # =====================
    # User Input
    # =====================

    user_input: str

    # =====================
    # Perception
    # =====================

    speech_text: str

    emotion: dict

    # AI 自身心情（预留，AI 心情系统使用）
    ai_mood: str

    # =====================
    # Conversation
    # =====================

    messages: List[BaseMessage]

    # =====================
    # Memory Context
    # =====================

    retrieved_memory: RetrievedMemory

    # =====================
    # Context
    # =====================

    system_context: str

    # =====================
    # Tool Calling（状态B）
    # =====================

    # planner 判断是否需要工具
    need_tool: bool

    # 工具名称（如 "tavily_search"）
    tool_name: str

    # 工具执行结果（注入 context_builder）
    tool_result: str

    # =====================
    # Planning
    # =====================

    plan: dict

    # =====================
    # LLM Request
    # =====================

    chat_request: ChatRequest

    # =====================
    # Response
    # =====================

    response: str

    # LLM流式输出
    response_chunks: list[StreamChunk]

    # =====================
    # Memory Pipeline
    # =====================

    memory_candidates: List[MemoryCandidate]

    accepted_memory: List[MemoryCandidate]