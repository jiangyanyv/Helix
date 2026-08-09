from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class MemoryCandidate:
    """
    候选记忆

    Extractor生成

    Judge判断

    Updater保存
    """


    # 记忆类型
    #
    # profile
    # preference
    # relationship
    # semantic
    # episodic
    # emotion
    memory_type: str


    # 记忆内容
    content: str


    # 标签
    tags: List[str] = field(
        default_factory=list
    )


    # 重要程度
    #
    # 后续由LLM Judge生成
    importance: float = 0.0


    # 扩展信息
    #
    # 例如：
    # 时间
    # 来源
    # 原始文本
    metadata: Dict = field(
        default_factory=dict
    )