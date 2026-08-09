from services.memory.emotion_service import (
    EmotionService
)

from services.memory.episodic_service import (
    EpisodicService
)

from services.memory.preference_service import (
    PreferenceService
)

from services.memory.profile_service import (
    ProfileService
)

from services.memory.relationship_service import (
    RelationshipService
)

from services.memory.semantic_service import (
    SemanticService
)


from memory.retrieved_memory import (
    RetrievedMemory
)



class MemoryManager:
    """
    Memory统一入口

    Node只依赖这里
    """


    def __init__(self):

        self.emotion = EmotionService()

        self.episodic = EpisodicService()

        self.preference = PreferenceService()

        self.profile = ProfileService()

        self.relationship = RelationshipService()

        self.semantic = SemanticService()



    def retrieve(
            self,
            user_id: str,
            query: str
    ) -> RetrievedMemory:


        return RetrievedMemory(

            profile=
            self.profile.get_profile(
                user_id
            ),


            preference=
            self.preference.retrieve(
                user_id
            ),


            episodic=
            self.episodic.search(
                user_id,
                query
            ),


            relationship=
            self.relationship.get(
                user_id
            ),


            semantic=
            self.semantic.search(
                query
            ),

        )