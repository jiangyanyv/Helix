from memory.retrieved_memory import RetrievedMemory

from services.container import container


class ContextBuilder:
    """
    构建当前对话需要的上下文

    Conversation Graph
            |
            v
      ContextBuilder
            |
            v
      RetrievedMemory
    """

    def __init__(self):

        self.profile_service = (
            container.profile_service
        )

        self.preference_service = (
            container.preference_service
        )

        self.relationship_service = (
            container.relationship_service
        )

        self.semantic_service = (
            container.semantic_service
        )

        self.episodic_service = (
            container.episodic_service
        )

        self.emotion_service = (
            container.emotion_service
        )


    def build(
            self,
            session_id: str,
            query: str
    ) -> RetrievedMemory:
        """
        根据当前用户和输入
        构建Memory Context
        """

        return RetrievedMemory(

            profile=self.profile_service.get(
                session_id
            ),

            preference=self.preference_service.get(
                session_id
            ),

            relationship=self.relationship_service.get(
                session_id
            ),

            semantic=self.semantic_service.search(
                query
            ),

            episodic=self.episodic_service.search(
                query
            ),

            emotion=self.emotion_service.get_summary(
                session_id
            )

        )