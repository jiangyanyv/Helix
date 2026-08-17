"""语音识别模块的数据模型。

包含情绪、语种枚举以及统一的语音输入结构 VoiceInput。
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EmotionTag(str, Enum):
    """SenseVoice 输出的情绪标签。"""

    UNKNOWN = "unknown"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"


class Language(str, Enum):
    """SenseVoice 输出的语种标签。"""

    UNKNOWN = "unknown"
    ZH = "zh"
    EN = "en"
    JA = "ja"
    KO = "ko"
    YUE = "yue"


class VoiceInput(BaseModel):
    """一段语音识别后的结构化结果。

    text        纯文本（已剥离 SenseVoice 标签，BGM 事件置空）
    emotion     情绪标签
    language    语种
    confidence  置信度 0~1（无有效文本时为 0）
    duration    该段音频时长（秒）
    timestamp   识别时间戳
    is_final    是否为最终结果（流式分段默认 True）
    """

    text: str = ""
    emotion: EmotionTag = EmotionTag.UNKNOWN
    language: Language = Language.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    duration: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    is_final: bool = True
