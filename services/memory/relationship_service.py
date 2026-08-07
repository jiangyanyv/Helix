# class RelationshipService:
#     """
#     用户关系状态服务
#
#     保存：
#
#     - 信任程度
#     - 熟悉程度
#     - 对话阶段
#     - 共同经历摘要
#
#     """
#
#     def __init__(self):
#
#         self.storage = {}
#
#
#     def get(
#             self,
#             session_id: str
#     ) -> dict:
#
#         return self.storage.get(
#             session_id,
#             {}
#         )
#
#
#     def update(
#             self,
#             session_id: str,
#             relationship: dict
#     ):
#
#         self.storage[session_id] = relationship

class RelationshipService:


    def get(
            self,
            session_id
    ):

        return {


            "relationship":

            "长期交流伙伴"


        }