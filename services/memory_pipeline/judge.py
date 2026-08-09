from typing import List

from memory.memory_candidate import MemoryCandidate



class MemoryJudge:
    """
    判断候选记忆是否进入长期存储

    后续：
    使用LLM进行判断

    当前：
    基础规则框架
    """


    def judge(
            self,
            candidates: List[MemoryCandidate]
    ) -> List[MemoryCandidate]:

        accepted = []


        for candidate in candidates:

            if self.is_worth_saving(
                    candidate
            ):

                accepted.append(
                    candidate
                )


        return accepted



    def is_worth_saving(
            self,
            candidate: MemoryCandidate
    ) -> bool:

        """
        判断记忆价值

        当前简单规则：

        """

        if not candidate.content:

            return False


        # 后续这里改LLM判断

        return True