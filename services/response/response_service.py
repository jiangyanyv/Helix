from services.llm.stream_processor import (
    StreamProcessor
)


class ResponseService:

    def __init__(

        self,

        llm_client

    ):

        self.llm = llm_client

        self.processor = StreamProcessor()


    def stream(

        self,

        request

    ):

        token_stream = self.llm.stream(

            request

        )

        yield from self.processor.process(

            token_stream

        )


    def generate(

        self,

        request

    ):

        chunks = []

        for chunk in self.stream(

                request

        ):

            chunks.append(chunk.text)

        return "".join(chunks)