from time import time


class EpisodicService:
    """
    情景记忆服务

    保存：
    用户经历过的事情（按 session + 时间戳）
    """

    def __init__(self):
        # session_id -> list[event_dict]
        self._storage: dict[str, list[dict]] = {}

    def search(
            self,
            session_id: str,
            query: str | None = None,
            top_k: int = 5
    ) -> list:
        """
        检索情景记忆。

        TODO:
        embedding search
        当前：返回最近 top_k 条
        """

        if session_id not in self._storage:
            # 返回默认种子数据（调试友好）
            return [
                {
                    "event": "用户正在研究ai技术",
                    "time": "最近"
                }
            ]

        events = list(reversed(self._storage[session_id]))
        return events[:top_k]

    def add(
            self,
            event: dict
    ):
        """
        添加情景记忆。

        期望 event 中包含：
        - session_id: str
        - content / event: str
        - tags: list (可选)
        - metadata: dict (可选)
        """

        session_id = event.pop("session_id", None)
        if not session_id:
            raise ValueError(
                "EpisodicService.add 要求 event 必须包含 session_id"
            )

        if session_id not in self._storage:
            self._storage[session_id] = []

        # 若未显式带 time，自动添加本地时间戳
        if "timestamp" not in event:
            event["timestamp"] = time()

        self._storage[session_id].append(event)

    def clear(
            self,
            session_id: str
    ):
        if session_id in self._storage:
            del self._storage[session_id]
