from core.graph import graph
from langchain_core.messages import HumanMessage

'''
流式输出
'''

class Agent:

    def __init__(self):
        self.graph = graph

    def chat(self, user_input: str):
        state = {
            "user_input": user_input,

            "messages": [
                HumanMessage(content=user_input)
            ]
        }

        result = self.graph.invoke(state)

        return result["response"]