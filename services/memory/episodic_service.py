# class EpisodicService:
#     """
#     情景记忆服务
#
#     保存：
#
#     用户经历过的事情
#
#     """
#
#     def __init__(self):
#
#         self.storage = []
#
#
#     def search(
#             self,
#             query: str,
#             top_k: int = 5
#     ) -> list:
#
#         # TODO:
#         # embedding search
#
#         return self.storage[:top_k]
#
#
#     def add(
#             self,
#             event: dict
#     ):
#
#         self.storage.append(
#             event
#         )

class EpisodicService:


    def search(
            self,
            session_id,
            query
    ):


        return [

            {

            "event":
            "用户正在开发陪伴式AI Agent",

            "time":
            "最近"

            }

        ]