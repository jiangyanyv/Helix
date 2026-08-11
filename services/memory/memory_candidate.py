from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Annotated

from pydantic import BaseModel, ConfigDict, Field, BeforeValidator


# =========================================================
# Emotion
# =========================================================

def _normalize_emotion(value: Any) -> Any:
    """
    兼容 LLM 两种输出：

    1.
    {
        "type": "伤感",
        "intensity": 0.8,
        "subject": "user"
    }

    2.
    "伤感"

    第二种属于模型输出不规范，这里自动转换，
    避免整个 MemoryExtractor 因为一个字段失败。
    """
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        return {
            "type": value,
            "intensity": 0.5,
            "subject": "user",
        }

    return value


# =========================================================
# Memory Type
# =========================================================

class MemoryType(str, enum.Enum):
    """3 类长期记忆类型。"""

    PROFILE = "profile"
    RELATIONSHIP = "relationship"
    EPISODIC = "episodic"


# =========================================================
# LLM Structured Output Schema
# =========================================================

class ProfileMetadataSchema(BaseModel):
    """PROFILE 类型的 LLM 输出结构。"""

    model_config = ConfigDict(extra="ignore")

    patch: Dict[str, Any] = Field(default_factory=dict)

    replace: bool = False


class RelationshipMetadataSchema(BaseModel):
    """
    RELATIONSHIP 类型的 LLM 输出结构。

    注意：
    不再让 LLM 输出 person_id。

    person_id 属于数据库实体层，
    后续由 RelationshipService.resolve_name()
    根据 canonical_name / aliases 解析。
    """

    model_config = ConfigDict(extra="ignore")

    canonical_name: str = ""

    aliases: List[str] = Field(
        default_factory=list
    )

    relation: str = "unknown"

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    extra: Dict[str, Any] = Field(
        default_factory=dict
    )


class EmotionSchema(BaseModel):
    """事件情绪结构。"""

    model_config = ConfigDict(extra="ignore")

    type: str = ""

    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    subject: str = "user"


class EpisodicMetadataSchema(BaseModel):
    """
    EPISODIC 类型的 LLM 输出结构。

    person_names：
        LLM 只输出人物名称。

    person_ids：
        不再由 LLM 提供。
        person_id 后续由 MemoryUpdater +
        RelationshipService 解析。
    """

    model_config = ConfigDict(extra="ignore")

    person_names: List[str] = Field(
        default_factory=list
    )

    emotion: Optional[
        Annotated[
            EmotionSchema,
            BeforeValidator(_normalize_emotion),
        ]
    ] = None

    timestamp: Optional[str] = None

    source: str = "conversation"


class MemoryCandidateSchema(BaseModel):
    """
    LLM 输出的单条候选记忆。

    LLM JSON
        ↓
    Pydantic 校验
        ↓
    MemoryCandidate
    """

    model_config = ConfigDict(extra="ignore")

    memory_type: MemoryType

    content: str = ""

    tags: List[str] = Field(
        default_factory=list
    )

    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    profile: Optional[
        ProfileMetadataSchema
    ] = None

    relationship: Optional[
        RelationshipMetadataSchema
    ] = None

    episodic: Optional[
        EpisodicMetadataSchema
    ] = None


class MemoryExtractionResult(BaseModel):
    """LLM 整体输出结构。"""

    model_config = ConfigDict(extra="ignore")

    memories: List[
        MemoryCandidateSchema
    ] = Field(
        default_factory=list
    )


# =========================================================
# Business Object
# =========================================================

@dataclass
class MemoryCandidate:
    """
    候选记忆：

        Extractor
            ↓
        Judge
            ↓
        Updater
            ↓
        Memory Service
    """

    memory_type: MemoryType

    content: str

    tags: List[str] = field(
        default_factory=list
    )

    importance: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def is_profile(self) -> bool:
        return self.memory_type is MemoryType.PROFILE

    @property
    def is_relationship(self) -> bool:
        return self.memory_type is MemoryType.RELATIONSHIP

    @property
    def is_episodic(self) -> bool:
        return self.memory_type is MemoryType.EPISODIC

    def meta(
        self,
        key: str,
        default: Any = None,
    ) -> Optional[Any]:
        return self.metadata.get(
            key,
            default,
        )