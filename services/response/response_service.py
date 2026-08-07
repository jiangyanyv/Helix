from services.llm.chat_request import (
    ChatRequest
)

from services.llm.client import (
    LLMClient
)



class ResponseService:


    def __init__(
            self,
            llm_client: LLMClient
    ):

        self.llm = llm_client



    def stream(
            self,
            request: ChatRequest
    ):

        """
        Agent回复流

        """

        yield from self.llm.stream(
            request
        )



    def generate(
            self,
            request: ChatRequest
    ):

        result = ""

        for chunk in self.stream(
                request
        ):

            result += chunk


        return result