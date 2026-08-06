from core.state import AgentState


def llm_node(state: AgentState):

    print("====== LLM ======")

    print("用户输入：", state["user_input"])
    print("Perception：", state["intent"])
    print("Perception：", state["emotion"])
    print("Perception：", state["entities"])
    print("Memory：", state["memories"])
    print("Planner：", state["strategy"])
    print("LLM：", state["response"] if "response" in state else "empty")

    return {
        "response": "你好，我是第一版陪伴Agent。"
    }