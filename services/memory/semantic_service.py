class SemanticService:
    """
    语义记忆检索服务

    后续连接：
    FAISS / Milvus / Chroma / pgvector

    当前：
    内存 list 实现
    """

    def __init__(self):
        self._storage: list[dict] = []

    def search(
            self,
            query: str,
            top_k: int = 5
    ) -> list:
        """
        根据语义检索事实记忆。

        TODO:
        embedding + vector search
        当前：简单返回最近 top_k 条
        """

        # 若有默认种子数据且 storage 为空，先返回种子
        if not self._storage:
            return [
                "LangGraph Agent架构",
                "Memory System设计"
            ]

        # 倒序返回最近 top_k 条
        items = [
            item.get("content", str(item))
            for item in reversed(self._storage)
        ]
        return items[:top_k]

    def add(
            self,
            memory: dict
    ):
        """添加一条语义记忆"""

        self._storage.append(memory)

    def clear(self):
        self._storage.clear()
