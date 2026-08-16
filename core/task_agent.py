"""Task Agent：工具调用子 Graph。

职责：
    接收 planner 的 tool_name + user_input，
    执行对应工具，返回格式化结果。

流程：
    tool_executor → result_formatter → END

输入：
    {user_input, tool_name}

输出：
    {tool_result}

当前支持的工具：
    - tavily_search: 网络搜索

未来扩展：
    - 加新工具只需在 TOOL_REGISTRY 注册
    - 需要多步骤时可加更多节点
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import StateGraph, END
from typing import TypedDict
from loguru import logger

from services.tools.tavily_search import tavily_search


# ============================================================
# State
# ============================================================

class TaskAgentState(TypedDict, total=False):
    """Task Agent 内部状态。"""

    user_input: str
    tool_name: str
    tool_result: str


# ============================================================
# 工具注册表
# ============================================================

def _run_tavily_search(query: str) -> str:
    """执行 Tavily 搜索。"""

    if not tavily_search.available:
        return ""
    return tavily_search.search(query)


# 工具名 → 执行函数
TOOL_REGISTRY = {
    "tavily_search": _run_tavily_search,
}


# ============================================================
# Nodes
# ============================================================

def tool_executor_node(state: TaskAgentState):
    """根据 tool_name 执行对应工具。"""

    tool_name = state.get("tool_name", "")
    user_input = state.get("user_input", "")

    executor = TOOL_REGISTRY.get(tool_name)

    if executor is None:
        logger.warning(
            f"[task_agent] 未知工具: {tool_name}"
        )
        return {"tool_result": ""}

    logger.info(
        f"[task_agent] 执行工具 | "
        f"tool={tool_name} | "
        f"query={user_input[:50]}"
    )

    try:
        result = executor(user_input)
    except Exception as e:  # noqa: BLE001
        logger.exception(
            f"[task_agent] 工具执行失败 | "
            f"tool={tool_name}, error={e}"
        )
        result = ""

    return {"tool_result": result}


def result_formatter_node(state: TaskAgentState):
    """格式化工具结果（当前直接透传，未来可加摘要/裁剪）。"""

    tool_result = state.get("tool_result", "")

    if not tool_result:
        tool_result = "（未获取到搜索结果）"

    return {"tool_result": tool_result}


# ============================================================
# Graph
# ============================================================

def build_task_agent():
    """构建 Task Agent 子 Graph。"""

    builder = StateGraph(TaskAgentState)

    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("result_formatter", result_formatter_node)

    builder.set_entry_point("tool_executor")
    builder.add_edge("tool_executor", "result_formatter")
    builder.add_edge("result_formatter", END)

    return builder.compile()


# 编译后的子 Graph（可直接作为节点加入主 Graph）
task_agent = build_task_agent()
