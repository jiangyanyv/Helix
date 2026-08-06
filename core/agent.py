from core.conversation_graph import graph
from llm.client import LLMClient
from core.session.manager import ConversationManager


class Agent:


    def __init__(self):

        self.graph = graph

        self.session_manager = ConversationManager()

        self.llm = LLMClient()



    def stream_chat(
            self,
            session_id: str,
            user_input: str
    ):
        # 保存用户输入

        self.session_manager.add_user_message(
            session_id,
            user_input
        )

        messages = self.session_manager.get_messages(
            session_id
        )

        # print(messages)


        # 运行Graph

        result = self.graph.invoke(
            {
                "session_id": session_id,

                "user_input": user_input,

                "messages": messages
            }
        )

        prompt = result["prompt"]

        # print("+++++++++++")
        # print(prompt)
        # print("+++++++++++")

        full_response = ""

        # LLM流式输出

        for token in self.llm.stream_chat(prompt):
            full_response += token

            yield token

        # 保存完整回复

        self.session_manager.add_ai_message(
            session_id,
            full_response
        )