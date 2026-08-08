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
            session_id: str
    ) -> dict:
        """获取关系状态，无数据则返回默认模板"""

        if session_id in self._storage:
            return self._storage[session_id]

        return {
            "relationship": "长期交流伙伴",
            "trust_level": 0.5,
            "familiarity": 0.3,
        }

    def update(
            self,
            session_id: str,
            relationship: dict
    ):
        """更新关系状态（增量合并）"""

        if session_id not in self._storage:
            self._storage[session_id] = {}

        self._storage[session_id].update(relationship)

    def clear(
            self,
            session_id: str
    ):
        if session_id in self._storage:
            del self._storage[session_id]
