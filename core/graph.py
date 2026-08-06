from langgraph.graph import StateGraph
from langgraph.graph import START, END

from core.state import AgentState

from core.nodes.perception import perception_node
from core.nodes.memory import memory_node
from core.nodes.planner import planner_node
from core.nodes.llm import llm_node


builder = StateGraph(AgentState)

builder.add_node("perception", perception_node)
builder.add_node("memory", memory_node)
builder.add_node("planner", planner_node)
builder.add_node("llm", llm_node)

builder.add_edge(START, "perception")
builder.add_edge("perception", "memory")
builder.add_edge("memory", "planner")
builder.add_edge("planner", "llm")
builder.add_edge("llm", END)

graph = builder.compile()