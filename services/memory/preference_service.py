class PreferenceService:
    """
    用户偏好服务

    保存：
    - 回复风格
    - 兴趣
    - 交互习惯
    - 长期偏好

    存储方式：
    JSON / Key-Value
    """

    def __init__(self):

        self.storage = {}


    def get(
            self,
            session_id: str
    ) -> dict:
        """
        获取用户偏好
        """

        return self.storage.get(
            session_id,
            {}
        )


    def update(
            self,
            session_id: str,
            preference: dict
    ):
        """
        更新用户偏好
        """

        self.storage[session_id] = preference