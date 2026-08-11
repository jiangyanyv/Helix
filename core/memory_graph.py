from langgraph.graph import StateGraph
from langgraph.graph import START
from langgraph.graph import END

from core.state import AgentState

from core.nodes.memory_extractor_node import memory_extractor_node
from core.nodes.memory_judge_node import memory_judge_node
from core.nodes.memory_updater_node import memory_updater_node

def build_graph():

    builder = StateGraph(AgentState)

    # 注册Node
    builder.add_node("extractor", memory_extractor_node)
    builder.add_node("judge", memory_judge_node)
    builder.add_node("updater", memory_updater_node)

    # 流程
    builder.add_edge(START, "extractor")
    builder.add_edge("extractor", "judge")
    builder.add_edge("judge", "updater")
    builder.add_edge("updater", END)

    return builder.compile()

memory_graph = build_graph()