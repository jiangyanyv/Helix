from core.state import AgentState


def perception_node(state: AgentState):

    print("====== Perception Center ======")

    return {
        "intent": "chat",
        "emotion": "neutral",
        "entities": []
    }