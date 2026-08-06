from core.state import AgentState


def llm_node(state: AgentState):

    print("====== LLM ======")

    print("用户输入：", state["user_input"])

    return {
        "response": "你好，我是第一版陪伴Agent。"
    }