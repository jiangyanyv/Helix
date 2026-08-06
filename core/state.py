from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage

from memory.retrieved_memory import RetrievedMemory

from memory.candidate import MemoryCandidate



class AgentState(TypedDict, total=False):
    """
    LangGraph 全局状态

    所有Node共享

    """

    # =====================
    # Session
    # =====================

    session_id: str


    # =====================
    # User Input
    # =====================

    user_input: str


    # =====================
    # Perception
    # =====================

    # 语音识别后的文本

    speech_text: str


    # SenseVoice等模型输出

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
    # Planning
    # =====================

    plan: dict


    # =====================
    # Prompt
    # =====================

    prompt: str


    # =====================
    # LLM
    # =====================

    response: str


    # =====================
    # Streaming
    # =====================

    response_chunks: List[str]


    # =====================
    # Memory Pipeline
    # =====================

    memory_candidates: List[MemoryCandidate]


    accepted_memory: List[MemoryCandidate]