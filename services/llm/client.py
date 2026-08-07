from abc import ABC
from abc import abstractmethod

from services.llm.chat_request import ChatRequest
from services.llm.chat_response import ChatResponse


class LLMClient(ABC):


    @abstractmethod
    def generate(
        self,
        request: ChatRequest
    ) -> ChatResponse:
        ...


    @abstractmethod
    def stream(
        self,
        request: ChatRequest
    ):
        ...