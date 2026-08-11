from langgraph.graph import (StateGraph,END)

from core.state import AgentState

from core.nodes.memory_retriever_node import memory_retriever_node

from core.nodes.context_builder_node import context_builder_node

from core.nodes.message_builder_node import message_builder_node

from core.nodes.response_generator_node import  response_generator_node



def build_graph():

    builder = StateGraph(AgentState)

    # 注册Node
    builder.add_node("memory_retriever",memory_retriever_node)
    builder.add_node("context_builder",context_builder_node)
    builder.add_node("message_builder",message_builder_node)
    builder.add_node("response_generator",response_generator_node)


    # 流程
    builder.set_entry_point("memory_retriever" )
    builder.add_edge("memory_retriever","context_builder")
    builder.add_edge("context_builder","message_builder")
    builder.add_edge("message_builder","response_generator")
    builder.add_edge("response_generator",END)


    return builder.compile()


conversation_graph = build_graph()