from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage

from services.llm.chat_request import ChatRequest
from memory.retrieved_memory import RetrievedMemory
from services.llm.stream_chunk import StreamChunk

from memory.candidate import MemoryCandidate



class AgentState(TypedDict, total=False):
    """
    LangGraph 全局状态

    所有Node共享

    """

    # =====================
    # Session & Turn
    # =====================

    session_id: str

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