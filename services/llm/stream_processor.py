from collections.abc import Iterable

from services.llm.stream_chunk import StreamChunk


class StreamProcessor:
    """
    将LLM Token流转换成适合TTS播放的Sentence Chunk

    功能：
        1. 根据句号、问号等切句
        2. 防止一句过长
        3. 防止一句过短
    """

    END_PUNCTUATION = {
        "。",
        "！",
        "？",
        "\n",
    }

    SOFT_PUNCTUATION = {
        "，",
        ",",
        "；",
        ";",
        "：",
        ":",
    }

    def __init__(
            self,
            min_chars: int = 15,
            max_chars: int = 80,
    ):
        """
        Args:
            min_chars:
                最短切分长度

            max_chars:
                超过这个长度即使没有句号也要切
        """

        self.min_chars = min_chars
        self.max_chars = max_chars

    def process(
            self,
            token_stream: Iterable[str]
    ):

        buffer = ""

        last_soft_index = -1

        for token in token_stream:

            buffer += token

            # 记录最近一次逗号位置
            if token in self.SOFT_PUNCTUATION:
                last_soft_index = len(buffer)

            # ------------------------
            # 第一优先级
            # 句号切句
            # ------------------------

            # 问号立即发送
            if token == "？":
                yield StreamChunk(
                    text=buffer,
                    is_sentence_end=True
                )

                buffer = ""
                last_soft_index = -1
                continue

            # 其它句号按长度判断
            if (
                    token in {
                "。",
                "！",
                "\n"
            }

                    and

                    len(buffer) >= self.min_chars
            ):
                yield StreamChunk(
                    text=buffer,
                    is_sentence_end=True
                )

                buffer = ""
                last_soft_index = -1
                continue

            # ------------------------
            # 第二优先级
            # 太长了
            # ------------------------

            if len(buffer) >= self.max_chars:

                # 优先从逗号处分割
                if last_soft_index > 0:

                    yield StreamChunk(
                        text=buffer[:last_soft_index],
                        is_sentence_end=False
                    )

                    buffer = buffer[last_soft_index:]

                    last_soft_index = -1

                else:

                    yield StreamChunk(
                        text=buffer,
                        is_sentence_end=False
                    )

                    buffer = ""

        # ------------------------
        # 收尾
        # ------------------------

        if buffer:

            yield StreamChunk(
                text=buffer,
                is_sentence_end=True
            )