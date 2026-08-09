from dataclasses import dataclass, field


@dataclass
class RetrievedMemory:
    """
    当前这一轮对话检索出的上下文
    """

    profile: dict = field(default_factory=dict)

    preference: dict = field(default_factory=dict)

    relationship: dict = field(default_factory=dict)

    semantic: list = field(default_factory=list)

    episodic: list = field(default_factory=list)

    emotion: dict = field(default_factory=dict)