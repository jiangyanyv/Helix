class RelationshipService:
    """
    用户社会关系状态服务

    保存：
    朋友、熟人、同事、客户、供应商等
    """

    def __init__(self):
        self._storage: dict[str, dict] = {}

    def get(
            self,
            user_id: str
    ) -> dict:
        """获取关系状态，无数据则返回默认模板"""

        if user_id in self._storage:
            return self._storage[user_id]

        return {

        }

    def update(
            self,
            user_id: str,
            relationship: dict
    ):
        """更新关系状态（增量合并）"""

        if user_id not in self._storage:
            self._storage[user_id] = {}

        self._storage[user_id].update(relationship)

    def clear(
            self,
            user_id: str
    ):
        if user_id in self._storage:
            del self._storage[user_id]
