from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RetrievedMemory:
    """
    单轮对话检索出的上下文记忆（简化为 3 类）。
    - profile      : 用户画像。合并原 Profile + Preference。identity/background/personality/preferences
    - relationships: 相关人物列表（RelationshipService.list_active 返回的完整 dicts）
    - episodic     : 相关事件列表（EpisodicService.search 结果，含 metadata.emotion/importance）
    """

    profile: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    episodic: List[Dict[str, Any]] = field(default_factory=list)
