from typing import Dict, List

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)


class ConversationManager:
    """
    会话管理器

    负责：
    1. 保存聊天历史
    2. 添加用户消息
    3. 添加AI回复
    4. 控制上下文长度
    """


    def __init__(self):

        # 当前用户所有session
        self.sessions: Dict[str, List[BaseMessage]] = {}


    def get_messages(
            self,
            session_id: str
    ) -> List[BaseMessage]:
        """
        获取历史消息
        """

        if session_id not in self.sessions:
            self.sessions[session_id] = []


        return self.sessions[session_id]



    def add_user_message(
            self,
            session_id: str,
            content: str
    ):

        messages = self.get_messages(session_id)


        messages.append(
            HumanMessage(
                content=content
            )
        )



    def add_ai_message(
            self,
            session_id: str,
            content: str
    ):

        messages = self.get_messages(session_id)


        messages.append(
            AIMessage(
                content=content
            )
        )



    def clear(
            self,
            session_id:str
    ):

        if session_id in self.sessions:
            del self.sessions[session_id]