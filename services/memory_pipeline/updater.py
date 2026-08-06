from typing import List

from services.container import container


class MemoryUpdater:
    """
    Memory写入服务

    根据MemoryCandidate类型

    路由到不同Memory Service

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



    def update(
            self,
            session_id: str,
            memories: List[MemoryCandidate]
    ):
        """
        保存通过Judge的记忆
        """

        for memory in memories:

            self._save_memory(
                session_id,
                memory
            )



    def _save_memory(
            self,
            session_id: str,
            memory: MemoryCandidate
    ):

        memory_type = memory.memory_type


        if memory_type == "profile":

            self.profile_service.update(

                session_id,

                {
                    "content": memory.content
                }

            )


        elif memory_type == "preference":

            self.preference_service.update(

                session_id,

                {
                    "content": memory.content,
                    "tags": memory.tags
                }

            )


        elif memory_type == "relationship":

            self.relationship_service.update(

                session_id,

                {
                    "content": memory.content
                }

            )


        elif memory_type == "semantic":

            self.semantic_service.add(

                {
                    "content": memory.content,
                    "tags": memory.tags,
                    "importance": memory.importance
                }

            )


        elif memory_type == "episodic":

            self.episodic_service.add(

                {
                    "content": memory.content,
                    "tags": memory.tags,
                    "metadata": memory.metadata
                }

            )


        elif memory_type == "emotion":

            self.emotion_service.add(

                session_id,

                {
                    "content": memory.content,
                    "metadata": memory.metadata
                }

            )