class ProfileService:
    """
    用户基础画像服务

    负责：
    - 获取用户基本信息

    不负责：
    - 记忆提取
    - 记忆更新
    - 数据库操作
    """

    def __init__(self):

        # 后续替换为数据库
        self.storage = {}


    def get(
            self,
            session_id: str
    ) -> dict:
        """
        获取用户画像
        """

        return self.storage.get(
            session_id,
            {}
        )


    def update(
            self,
            session_id: str,
            profile: dict
    ):
        """
        更新用户画像

        后续由 Memory Updater 调用
        """

        self.storage[session_id] = profile