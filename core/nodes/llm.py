from core.state import AgentState


def llm_node(state: AgentState):

    print("====== LLM ======")

    print(state["prompt"])

    return {
        "response": "你好，我是第一版陪伴Agent。"
    }