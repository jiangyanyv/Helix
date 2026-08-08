class PreferenceService:
    """
    用户偏好服务

    保存：
    - 回复风格
    - 兴趣
    - 交互习惯
    - 长期偏好

    存储方式：
    内存 K-V（后续替换为 JSON / Key-Value DB）
    """

    def __init__(self):
        self._storage: dict[str, dict] = {}

    def retrieve(
            self,
            session_id: str
    ) -> dict:
        """
        获取用户偏好。

        若无数据则返回默认模板。
        """

        if session_id in self._storage:
            return self._storage[session_id]

        return {
            "likes": ["AI技术", "游戏", "听歌"],
            "communication": "喜欢详细解释"
        }

    def update(
            self,
            session_id: str,
            preference: dict
    ):
        """
        更新用户偏好（增量合并）。
        """

        if session_id not in self._storage:
            self._storage[session_id] = {}

        # 针对 likes 做并集，避免覆盖原有兴趣
        old_likes = self._storage[session_id].get("likes", [])
        new_likes = preference.get("likes", [])
        if new_likes:
            merged_likes = list(dict.fromkeys(old_likes + new_likes))  # 保持顺序去重
            self._storage[session_id]["likes"] = merged_likes
            preference = {k: v for k, v in preference.items() if k != "likes"}

        self._storage[session_id].update(preference)

    def clear(
            self,
            session_id: str
    ):
        if session_id in self._storage:
            del self._storage[session_id]
