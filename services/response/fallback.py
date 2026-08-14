import random

from services.llm.stream_chunk import StreamChunk


# ============================================================
# 兜底回复文案
#
# LLM 调用失败时使用。
# 要求：
# - 符合陪伴式 AI 人设
# - 不暴露技术错误
# - 自然引导用户继续对话
# ============================================================

FALLBACK_RESPONSES = [
    "呀，我刚刚走神了，主人再说一次好不好？",
    "诶嘿，刚才好像有点没听清呢，能再讲一遍吗？",
    "呜……刚才脑子突然卡了一下，主人别介意呀。",
    "抱歉抱歉，刚才神游了一下，我们继续聊？",
    "嗯……我刚刚好像错过了一些，再说一次给我听好不好？",
    "呀，刚才那一下是怎么回事呢……主人再和我说说？",
    "诶，我刚才好像没接上话，不是故意的呀，再来一次？",
]


def pick_fallback() -> str:
    """随机选择一条兜底回复。"""
    return random.choice(FALLBACK_RESPONSES)


def create_fallback_chunk(turn_id: str | None) -> StreamChunk:
    """
    创建一条完整的兜底 StreamChunk。

    Args:
        turn_id: 当前对话轮次 ID。

    Returns:
        可直接进入 TTS / AudioQueue 的 StreamChunk。
    """
    return StreamChunk(
        text=pick_fallback(),
        turn_id=turn_id,
        is_sentence_end=True,
        interruptible=True,
    )