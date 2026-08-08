from langgraph.graph import (
    StateGraph,
    END
)


from core.state import AgentState



from core.nodes.memory_retriever import (
    memory_retriever_node
)


from core.nodes.context_builder import (
    context_builder_node
)


from core.nodes.message_builder import (
    message_builder_node
)


from core.nodes.response_generator import (
    response_generator_node
)



def build_graph():

    workflow = StateGraph(
        AgentState
    )


    # =====================
    # 注册Node
    # =====================


    workflow.add_node(
        "memory_retriever",
        memory_retriever_node
    )


    workflow.add_node(
        "context_builder",
        context_builder_node
    )


    workflow.add_node(
        "message_builder",
        message_builder_node
    )


    workflow.add_node(
        "response_generator",
        response_generator_node
    )



    # =====================
    # 流程
    # =====================


    workflow.set_entry_point(
        "memory_retriever"
    )


    workflow.add_edge(
        "memory_retriever",
        "context_builder"
    )


    workflow.add_edge(
        "context_builder",
        "message_builder"
    )


    workflow.add_edge(
        "message_builder",
        "response_generator"
    )


    workflow.add_edge(
        "response_generator",
        END
    )


    return workflow.compile()



conversation_graph = build_graph()