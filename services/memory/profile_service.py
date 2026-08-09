class ProfileService:
    """
    用户基础画像服务

    负责：
    - 获取用户基本信息
    - 更新用户画像

    存储方式：
    内存 K-V（后续替换为数据库）
    """

    def __init__(self):
        # 后续替换为数据库
        self._storage: dict[str, dict] = {}

    def get_profile(
            self,
            user_id: str
    ) -> dict:
        """
        获取用户画像。

        若尚未存储任何画像，则返回默认模板。
        """

        if user_id in self._storage:
            return self._storage[user_id]

        # 默认画像（首次访问时返回）
        return {
            "name": "江燕语",
            "job": "自由人",
            "personality": "喜欢深入理解技术原理"
        }

    def update(
            self,
            user_id: str,
            profile: dict
    ):
        """
        更新用户画像（增量合并）。

        后续由 Memory Updater 调用。
        """

        if user_id not in self._storage:
            self._storage[user_id] = {}

        # 增量合并，避免覆盖已有字段
        self._storage[user_id].update(profile)

    def clear(
            self,
            user_id: str
    ):
        """清空指定用户的画像（调试/重置用）"""

        if user_id in self._storage:
            del self._storage[user_id]
