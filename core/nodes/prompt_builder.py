from core.prompts.system_prompt import SYSTEM_PROMPT
from core.state import AgentState


def prompt_builder_node(
        state: AgentState
):
    """
    构建LLM Prompt

    输入：

    user_input

    retrieved_memory


    输出：

    prompt

    """

    print(
        "====== Prompt Builder ======"
    )


    memory = state["retrieved_memory"]


    profile_context = format_profile(
        memory.profile
    )


    preference_context = format_preference(
        memory.preference
    )


    relationship_context = format_relationship(
        memory.relationship
    )


    semantic_context = format_list_memory(
        memory.semantic
    )


    episodic_context = format_list_memory(
        memory.episodic
    )


    emotion_context = format_emotion(
        memory.emotion
    )


    prompt = f"""
{SYSTEM_PROMPT}

# 用户画像

{profile_context}


# 用户偏好

{preference_context}


# 关系状态

{relationship_context}


# 用户相关事实

{semantic_context}


# 用户过去经历

{episodic_context}


# 用户近期情绪

{emotion_context}


# 当前用户输入

{state["user_input"]}

"""

    # print(prompt)


    return {

        "prompt": prompt

    }



def format_profile(
        profile: dict
):
    if not profile:
        return "暂无用户信息"

    return str(profile)



def format_preference(
        preference: dict
):
    if not preference:
        return "暂无偏好信息"

    return str(preference)



def format_relationship(
        relationship: dict
):
    if not relationship:
        return "暂无关系信息"

    return str(relationship)



def format_list_memory(
        memories: list
):

    if not memories:
        return "暂无"

    return "\n".join(
        [
            str(item)
            for item in memories
        ]
    )



def format_emotion(
        emotion: dict
):

    if not emotion:
        return "暂无"

    return str(emotion)