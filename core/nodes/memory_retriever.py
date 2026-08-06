from core.state import AgentState


def memory_retriever_node(state: AgentState):

    print("====== Memory Retriever ======")


    memory = {

        "profile":[
            "用户是一名软件工程师"
        ],


        "personality":[
            "用户喜欢先理解架构再编码"
        ],


        "preference":[
            "用户喜欢详细技术解释"
        ],


        "recent_events":[
            "用户正在开发陪伴式AI项目"
        ]

    }


    return {
        "retrieved_memory": memory
    }