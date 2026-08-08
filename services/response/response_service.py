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

        request,

        turn_id: str | None = None,

    ):
        """
        流式生成回复。

        Args:
            request: ChatRequest 对象
            turn_id: 当前对话轮次 ID，用于关联 TTS 与 Turn 生命周期。
                     未传入时 Chunk.turn_id 为 None，
                     但仍会正常生成文本。
        """

        token_stream = self.llm.stream(

            request

        )

        yield from self.processor.process(

            token_stream,

            turn_id=turn_id,

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