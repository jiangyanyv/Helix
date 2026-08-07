from core.prompts.system_prompt import SYSTEM_PROMPT
from memory.retrieved_memory import RetrievedMemory



class ContextBuilder:
    """
    构建LLM运行上下文

    输入:
        RetrievedMemory

    输出:
        system_context

    注意:
        不负责查询Memory
        不负责LLM
    """



    def build(
            self,
            retrieved_memory: RetrievedMemory | None
    ) -> str:


        sections = []


        # =====================
        # Persona
        # =====================

        sections.append(
            self.build_persona()
        )


        # =====================
        # Memory
        # =====================

        if retrieved_memory:

            memory_context = (
                self.build_memory_context(
                    retrieved_memory
                )
            )

            if memory_context:

                sections.append(
                    memory_context
                )


        return "\n\n".join(
            sections
        )



    def build_persona(self) -> str:

        return SYSTEM_PROMPT



    def build_memory_context(
            self,
            memory: RetrievedMemory
    ) -> str:


        sections = []


        if memory.profile:

            sections.append(
                f"""
【用户基本信息】

{memory.profile}
""".strip()
            )


        if memory.preference:

            sections.append(
                f"""
【用户偏好】

{memory.preference}
""".strip()
            )


        if memory.episodic:

            sections.append(
                f"""
【近期事件】

{memory.episodic}
""".strip()
            )


        if memory.relationship:

            sections.append(
                f"""
【关系信息】

{memory.relationship}
""".strip()
            )


        return "\n\n".join(
            sections
        )