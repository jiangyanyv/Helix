"""Conversation Graph 构建。

状态A（ENABLE_TOOL_CALLING=false）：
    memory_retriever → context_builder → message_builder → response_generator

状态B（ENABLE_TOOL_CALLING=true）：
    planner → conditional_edge
      ├─ "normal" → memory_retriever → context_builder → ...
      └─ "tool"   → task_agent → context_builder → ...
    两条路径汇入 context_builder → message_builder → response_generator
"""

from langgraph.graph import StateGraph, END

from config import Config
from core.state import AgentState

from core.nodes.memory_retriever_node import memory_retriever_node
from core.nodes.context_builder_node import context_builder_node
from core.nodes.message_builder_node import message_builder_node
from core.nodes.response_generator_node import response_generator_node


def route_by_strategy(state: AgentState) -> str:
    """Conditional edge 路由：planner 之后决定走工具还是走正常链路。"""

    if state.get("need_tool"):
        return "tool"
    return "normal"


def build_graph():
    builder = StateGraph(AgentState)

    # 公共节点
    builder.add_node("memory_retriever", memory_retriever_node)
    builder.add_node("context_builder", context_builder_node)
    builder.add_node("message_builder", message_builder_node)
    builder.add_node("response_generator", response_generator_node)

    if Config.ENABLE_TOOL_CALLING:
        # =====================
        # 状态B：插入 planner + task_agent
        # =====================
        from core.nodes.planner_node import planner_node
        from core.task_agent import task_agent

        builder.add_node("planner", planner_node)
        builder.add_node("task_agent", task_agent)

        builder.set_entry_point("planner")

        # planner 之后根据 need_tool 路由
        builder.add_conditional_edges(
            "planner",
            route_by_strategy,
            {
                "normal": "memory_retriever",
                "tool": "task_agent",
            },
        )

        # task_agent 完成后进入 context_builder
        # builder.add_edge("task_agent", "context_builder")
        builder.add_edge("task_agent", "memory_retriever")

    else:
        # =====================
        # 状态A：当前方案
        # =====================
        builder.set_entry_point("memory_retriever")

    # 公共边
    builder.add_edge("memory_retriever", "context_builder")
    builder.add_edge("context_builder", "message_builder")
    builder.add_edge("message_builder", "response_generator")
    builder.add_edge("response_generator", END)

    return builder.compile()


conversation_graph = build_graph()
