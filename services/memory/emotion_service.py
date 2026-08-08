class EmotionService:
    """
    用户情绪轨迹服务

    用于：
    - 最近状态
    - 情绪趋势
    - Reflection
    """

    def __init__(self):
        # session_id -> list[emotion_dict]
        self._storage: dict[str, list[dict]] = {}

    def get(
            self,
            session_id: str
    ) -> dict:
        """
        获取用户当前情绪摘要。

        返回：
        {
            "current": str,
            "recent_history": list[最近N条]
        }
        """

        history = self._storage.get(session_id, [])
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
            session_id: str,
            emotion: dict
    ):
        """
        追加一条情绪记录。
        """

        if session_id not in self._storage:
            self._storage[session_id] = []

        self._storage[session_id].append(emotion)

    def get_summary(
            self,
            session_id: str
    ) -> dict:
        """等价于 get(session_id)，对外兼容接口"""
        return self.get(session_id)

    def clear(
            self,
            session_id: str
    ):
        if session_id in self._storage:
            del self._storage[session_id]
