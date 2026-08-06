# from langgraph.graph import StateGraph
# from langgraph.graph import START, END
#
# from core.state import AgentState
#
# from core.nodes.perception import perception_node
# from core.nodes.memory_retriever import memory_retriever_node
# from core.nodes.planner import planner_node
# from core.nodes.prompt_builder import prompt_builder_node
#
#
# builder = StateGraph(AgentState)
#
# builder.add_node("perception", perception_node)
# builder.add_node("memory_retriever", memory_retriever_node)
# builder.add_node("planner", planner_node)
# builder.add_node("prompt", prompt_builder_node)
#
# builder.add_edge(START, "perception")
# builder.add_edge("perception", "memory_retriever")
# builder.add_edge("memory_retriever", "planner")
# builder.add_edge("planner", "prompt")
# builder.add_edge("prompt", END)
#
# graph = builder.compile()


from langgraph.graph import StateGraph, END
from core.state import AgentState
from core.nodes.perception import perception_node
from core.nodes.memory_retriever import memory_retriever_node
from core.nodes.planner import planner_node
from core.nodes.prompt_builder import prompt_builder_node

builder = StateGraph(AgentState)

builder.add_node("perception", perception_node)
builder.add_node("memory_retriever", memory_retriever_node)
builder.add_node("planner", planner_node)
builder.add_node("prompt_builder", prompt_builder_node)

builder.set_entry_point("perception")

builder.add_edge("perception", "memory_retriever")
builder.add_edge("memory_retriever", "planner")
builder.add_edge("planner", "prompt_builder")
builder.add_edge("prompt_builder", END)


graph = builder.compile()


# from langgraph.graph import StateGraph, END
# from core.state import AgentState
# from core.nodes.perception import perception_node
# from core.nodes.memory_retriever import memory_retriever_node
# from core.nodes.planner import planner_node
# from core.nodes.prompt_builder import prompt_builder_node
# from core.nodes.llm_node import llm_node
#
# def build_agent_graph():
#     graph = StateGraph(AgentState)
#     graph.add_node("perception", perception_node)
#     graph.add_node("memory_retriever", memory_retriever_node)
#     graph.add_node("planner", planner_node)
#     graph.add_node("prompt_builder", prompt_builder_node)
#     graph.add_node("llm", llm_node)
#     graph.set_entry_point("perception")
#     graph.add_edge("perception", "memory_retriever")
#     graph.add_edge("memory_retriever", "planner")
#     graph.add_edge("planner", "prompt_builder")
#     graph.add_edge("prompt_builder", "llm")
#     graph.add_edge("llm", END)
#     return graph.compile()