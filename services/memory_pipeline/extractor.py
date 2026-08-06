from typing import List

from memory.candidate import MemoryCandidate



class MemoryExtractor:
    """
    从对话中提取候选记忆

    注意：

这里不判断是否保存。

只负责发现。

"""


    def extract(
            self,
            user_text: str,
            ai_text: str
    ) -> List[MemoryCandidate]:

        candidates = []


        # TODO:
        #
        # 后续替换：
        # LLM Structured Output
        #
        # 当前：
        # 返回空列表


        return candidates