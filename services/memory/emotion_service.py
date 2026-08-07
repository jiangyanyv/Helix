# class EmotionService:
#     """
#     用户情绪轨迹服务
#
#     用于：
#
#     - 最近状态
#     - 情绪趋势
#     - Reflection
#
#     """
#
#     def __init__(self):
#
#         self.storage = {}
#
#
#     def get_summary(
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
#     def add(
#             self,
#             session_id: str,
#             emotion: dict
#     ):
#
#         if session_id not in self.storage:
#
#             self.storage[session_id] = []
#
#
#         self.storage[session_id].append(
#             emotion
#         )

class EmotionService:


    def get(
            self,
            session_id
    ):


        return {


            "current":

            "normal"


        }