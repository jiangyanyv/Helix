from core.state import AgentState
from core.prompts.system_prompt import SYSTEM_PROMPT


def prompt_node(state: AgentState):

    print("====== Prompt Builder ======")

    prompt = f"""
{SYSTEM_PROMPT}

【用户输入】
{state["user_input"]}

【长期记忆】
{state["memories"]}

【回复策略】
{state["strategy"]}
"""

    return {
        "prompt": prompt
    }