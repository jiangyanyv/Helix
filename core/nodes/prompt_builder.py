from core.state import AgentState
from core.prompts.system_prompt import SYSTEM_PROMPT


def prompt_node(state: AgentState):


    print("====== Prompt Builder ======")


    memory = state["retrieved_memory"]


    prompt = f"""

{SYSTEM_PROMPT}


用户输入：

{state["user_input"]}


用户相关信息：

用户特点：
{memory["personality"]}


用户偏好：
{memory["preference"]}


近期事件：
{memory["recent_events"]}


回复策略：

{state["strategy"]}

"""


    return {
        "prompt": prompt
    }