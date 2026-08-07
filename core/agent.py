from core.conversation_graph import graph
from core.session.manager import ConversationManager


class Agent:
    """
    Agent入口

    负责:

    - session管理
    - 调用LangGraph
    - 返回结果


    不负责:

    - Prompt
    - Memory
    - LLM调用

    """


    def __init__(self):

        self.graph = graph

        self.session_manager = ConversationManager()



    def stream_chat(
            self,
            session_id: str,
            user_input: str
    ):


        # =====================
        # 保存用户消息
        # =====================

        self.session_manager.add_user_message(
            session_id,
            user_input
        )


        # =====================
        # 获取历史消息
        # =====================

        messages = (
            self.session_manager
            .get_messages(session_id)
        )


        # =====================
        # 运行Graph
        # =====================

        result = self.graph.invoke(
            {

                "session_id": session_id,

                "user_input": user_input,

                "messages": messages

            }
        )


        # =====================
        # 获取AI回复
        # =====================

        response = result.get(
            "response",
            ""
        )


        # 保存

        self.session_manager.add_ai_message(
            session_id,
            response
        )


        yield response