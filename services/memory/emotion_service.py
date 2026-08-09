class EmotionService:
    """
    用户情绪轨迹服务

    用于：
    - 最近状态
    - 情绪趋势
    - Reflection
    """

    def __init__(self):
        # user_id -> list[emotion_dict]
        self._storage: dict[str, list[dict]] = {}

    def get(
            self,
            user_id: str
    ) -> dict:
        """
        获取用户当前情绪摘要。

        返回：
        {
            "current": str,
            "recent_history": list[最近N条]
        }
        """

        history = self._storage.get(user_id, [])
        if not history:
            return {
                "current": "normal",
                "recent_history": []
            }

        latest = history[-1]
        return {
            "current": latest.get("content", "normal"),
            "recent_history": history[-5:]
        }

    def add(
            self,
            user_id: str,
            emotion: dict
    ):
        """
        追加一条情绪记录。
        """

        if user_id not in self._storage:
            self._storage[user_id] = []

        self._storage[user_id].append(emotion)

    def get_summary(
            self,
            user_id: str
    ) -> dict:
        """等价于 get(user_id)，对外兼容接口"""
        return self.get(user_id)

    def clear(
            self,
            user_id: str
    ):
        if user_id in self._storage:
            del self._storage[user_id]
