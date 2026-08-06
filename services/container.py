from services.memory.profile_service import ProfileService
from services.memory.preference_service import PreferenceService
from services.memory.relationship_service import RelationshipService
from services.memory.semantic_service import SemanticService
from services.memory.episodic_service import EpisodicService
from services.memory.emotion_service import EmotionService



class ServiceContainer:
    """
    全局Service容器

    管理：

    Memory相关Service生命周期

    保证：

    全局单例

    """


    def __init__(self):

        self.profile_service = ProfileService()

        self.preference_service = PreferenceService()

        self.relationship_service = RelationshipService()

        self.semantic_service = SemanticService()

        self.episodic_service = EpisodicService()

        self.emotion_service = EmotionService()



# 全局唯一实例

container = ServiceContainer()