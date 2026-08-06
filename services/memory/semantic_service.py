class SemanticService:
    """
    语义记忆检索服务

    后续连接：

    FAISS
    Milvus
    Chroma
    pgvector

    """

    def __init__(self):

        self.storage = []


    def search(
            self,
            query: str,
            top_k: int = 5
    ) -> list:
        """
        根据语义检索事实记忆
        """

        # TODO:
        # embedding + vector search

        return self.storage[:top_k]


    def add(
            self,
            memory: dict
    ):

        self.storage.append(
            memory
        )