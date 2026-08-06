from langgraph.graph import StateGraph
from langgraph.graph import START, END

from core.state import AgentState

from core.nodes.perception import perception_node
from core.nodes.memory_retriever import memory_retriever_node
from core.nodes.planner import planner_node
from core.nodes.prompt_builder import prompt_node
# from core.nodes.llm import llm_node


builder = StateGraph(AgentState)

builder.add_node("perception", perception_node)
builder.add_node("memory_retriever", memory_retriever_node)
builder.add_node("planner", planner_node)
builder.add_node("prompt", prompt_node)
# builder.add_node("llm", llm_node)

builder.add_edge(START, "perception")
builder.add_edge("perception", "memory_retriever")
builder.add_edge("memory_retriever", "planner")
builder.add_edge("planner", "prompt")
# builder.add_edge("prompt", "llm")
# builder.add_edge("llm", END)
builder.add_edge("prompt", END)

graph = builder.compile()