from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from services.llm.chat_request import ChatRequest
from services.llm.deepseek_client import DeepSeekClient
from services.memory.memory_candidate import (
    MemoryCandidate,
    MemoryType,
)


# ============================================================
# LLM Judge Schema
# ============================================================


class MemoryJudgeItem(BaseModel):
    """
    LLM 对单条候选记忆的判断结果。
    """

    model_config = ConfigDict(extra="ignore")

    index: int = Field(
        description="候选记忆在输入列表中的索引，从0开始"
    )

    worth_saving: bool = Field(
        description="是否值得进入长期记忆"
    )

    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="长期记忆重要程度，0~1"
    )

    reason: str = Field(
        default="",
        description="简短判断理由"
    )


class MemoryJudgeResult(BaseModel):
    """
    MemoryJudge 的整体结构化输出。
    """

    model_config = ConfigDict(extra="ignore")

    results: List[MemoryJudgeItem] = Field(
        default_factory=list
    )


# ============================================================
# MemoryJudge
# ============================================================


class MemoryJudge:
    """
    判断候选记忆是否值得进入长期存储。

    流程：

        MemoryCandidate
              ↓
            LLM
              ↓
        MemoryJudgeResult
              ↓
        accepted candidates

    Judge 只负责：

    1. 是否值得保存
    2. importance 评分

    不负责：

    - 修改 memory_type
    - 修改 content
    - 修改 metadata
    - 写数据库
    """

    SYSTEM_PROMPT = """
你是一个“长期记忆价值判断器”。

你的任务是判断 MemoryExtractor 提取出来的候选记忆，
是否值得进入用户的长期记忆系统。

你不是 MemoryExtractor。

候选记忆已经由上游 Extractor 提取完成，
你只负责判断：

1. 是否值得长期保存
2. 给出 0~1 的 importance

不要修改候选记忆本身。

====================
一、总体原则
====================

只有“未来再次聊天时可能有帮助”的信息，
才值得进入长期记忆。

1. 用户明确讲述的个人经历、体验、事件或记忆；
2. 内容具体，而不是纯粹的无意义闲聊；
3. 未来再次谈论时具有一定上下文价值；
原则上应该 worth_saving = true。
特别注意：

梦境、旅行经历、游戏经历、童年经历、与朋友发生的事情、
用户提到的特殊体验等，即使重要程度较低，也可以保存为 Episodic。

普通闲聊、临时状态、一次性想法、
没有持续价值的信息，应当拒绝保存。

====================
二、PROFILE 判断标准
====================

PROFILE 描述用户本人。

优先保存：

- 长期身份
- 职业
- 学习方向
- 长期目标
- 稳定兴趣
- 稳定偏好
- 性格特点
- 长期习惯
- 沟通偏好
- 重要背景

例如：

“我是 Java 后端开发，现在准备转 AI Agent。”

值得保存。

例如：

“我今天晚上想吃火锅。”

通常不值得保存。

PROFILE 如果属于稳定、长期的信息，
importance 通常为 0.7~1.0。

====================
三、RELATIONSHIP 判断标准
====================

RELATIONSHIP 描述用户与现实人物之间的关系。

优先保存：

- 家人
- 朋友
- 同事
- 同学
- 重要认识的人
- 经常提及的人
- 与用户存在长期关系的人

例如：

“老王是我大学时期认识的朋友。”

值得保存。

如果只是新闻、网络人物、
小说人物、游戏角色，
除非明确属于用户长期关系，
否则不要保存为 relationship。

RELATIONSHIP 如果关系明确，
importance 通常为 0.7~1.0。

====================
四、EPISODIC 判断标准
====================

EPISODIC 是具体事件或经历。

优先保存：

- 重要人生经历
- 游戏成就
- 项目经历
- 工作经历
- 重要决定
- 重要计划
- 特殊事件
- 与重要人物有关的事件
- 以后再次提到时具有明显回忆价值的事件

例如：

“我终于打上超凡大师了。”

值得保存。

例如：

“今天中午吃了米饭。”

通常不值得保存。

例如：

“我准备辞职转行做 AI Agent。”

如果这是用户的重要长期决定，值得保存。
另外，只要是用户明确讲述的个人经历，并且不是明显无意义的瞬时信息，就允许进入 Episodic。

====================
五、情绪信息
====================

单纯的临时情绪通常不足以成为长期记忆。

例如：

“今天好烦。”

通常不值得保存。

但是如果情绪与重要事件绑定：

“老王现在联系不上了，大家都很想念他。”

这是一个具有长期回忆价值的事件，
可以保存为 EPISODIC。

====================
六、importance 评分
====================

0.0 ~ 0.2：

几乎没有长期价值。

普通闲聊、临时状态。

0.3 ~ 0.4：

价值较低。

可能未来偶尔有用。

0.5 ~ 0.6：

中等价值。

未来聊天可能有一定帮助。

0.7 ~ 0.8：

较高价值。

明显值得长期保存。

0.9 ~ 1.0：

非常重要。

核心个人信息、重大人生事件、
非常重要的人际关系、重大成就等。

====================
七、重要限制
====================

1. 不要因为候选已经存在就默认保存。

2. 不要为了凑数量而保存。

3. 不要修改候选的 content。

4. 不要修改候选的 metadata。

5. 不要改变 memory_type。

6. 只能根据候选本身判断。

7. 如果信息明显是临时性的，应拒绝。

8. 如果信息未来再次聊天时具有明显价值，应倾向保存。

====================
八、输出格式
====================

只能输出 JSON。

格式：

{
    "results": [
        {
            "index": 0,
            "worth_saving": true,
            "importance": 0.85,
            "reason": "这是用户与重要人物之间的长期关系"
        }
    ]
}

index 必须对应输入候选的索引。

每个候选都应该有一个判断结果。
""".strip()

    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
    ) -> None:
        self.llm = llm_client or DeepSeekClient()

    # =========================================================
    # Public
    # =========================================================

    def judge(
        self,
        candidates: List[MemoryCandidate],
    ) -> List[MemoryCandidate]:

        if not candidates:
            return []

        try:
            prompt = self._build_prompt(candidates)

            request = ChatRequest(
                messages=[
                    SystemMessage(
                        content=self.SYSTEM_PROMPT
                    ),
                    HumanMessage(
                        content=prompt
                    ),
                ],
                model=self.llm.model,
                stream=False,
                temperature=0.0,
                top_p=1.0,
                max_tokens=2000,
            )

            response = self.llm.generate(request)

            raw = (response.text or "").strip()

            if not raw:
                logger.warning(
                    "[MemoryJudge] LLM 返回为空"
                )

                return self._fallback(candidates)

            logger.debug(
                "[MemoryJudge] LLM raw response: "
                f"{raw[:1500]}"
            )

            result = self._parse_result(raw)

            return self._apply_result(
                candidates,
                result,
            )

        except Exception as e:
            logger.exception(
                f"[MemoryJudge] 判断失败: {e}"
            )

            # Judge 失败时不要让整个 Memory Graph 崩掉。
            # 这里采用保守策略：
            # 仅保留原本 importance 较高的候选。
            return self._fallback(candidates)

    # =========================================================
    # Prompt
    # =========================================================

    @staticmethod
    def _build_prompt(
        candidates: List[MemoryCandidate],
    ) -> str:

        blocks = []

        for index, candidate in enumerate(candidates):

            blocks.append(
                f"""
【候选 {index}】

memory_type:
{candidate.memory_type.value}

content:
{candidate.content}

tags:
{candidate.tags}

importance_initial:
{candidate.importance}

metadata:
{candidate.metadata}
""".strip()
            )

        return (
            "请判断下面所有候选记忆是否值得进入长期记忆。\n\n"
            + "\n\n".join(blocks)
            + "\n\n请严格按照系统要求输出 JSON。"
        )

    # =========================================================
    # Parse
    # =========================================================

    @staticmethod
    def _parse_result(
        raw: str,
    ) -> MemoryJudgeResult:

        import json
        import re

        cleaned = raw.strip()

        # 兼容 ```json
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)

        except json.JSONDecodeError:

            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start < 0 or end <= start:
                raise ValueError(
                    "LLM 返回内容中没有有效 JSON"
                )

            data = json.loads(
                cleaned[start:end + 1]
            )

        return MemoryJudgeResult.model_validate(
            data
        )

    # =========================================================
    # Apply
    # =========================================================

    @staticmethod
    def _apply_result(
        candidates: List[MemoryCandidate],
        result: MemoryJudgeResult,
    ) -> List[MemoryCandidate]:

        accepted: List[MemoryCandidate] = []

        result_map = {
            item.index: item
            for item in result.results
        }

        for index, candidate in enumerate(candidates):

            decision = result_map.get(index)

            if decision is None:
                logger.warning(
                    f"[MemoryJudge] 缺少候选 {index} 的判断结果，跳过"
                )
                continue

            importance = max(
                0.0,
                min(
                    1.0,
                    float(decision.importance)
                )
            )

            candidate.importance = importance

            if not decision.worth_saving:

                logger.debug(
                    "[MemoryJudge] 拒绝 | "
                    f"type={candidate.memory_type.value} "
                    f"content={candidate.content!r} "
                    f"importance={importance:.2f} "
                    f"reason={decision.reason}"
                )

                continue

            accepted.append(candidate)

            logger.info(
                "[MemoryJudge] 接受 | "
                f"type={candidate.memory_type.value} "
                f"content={candidate.content!r} "
                f"importance={importance:.2f}"
            )

        return accepted

    # =========================================================
    # Fallback
    # =========================================================

    @staticmethod
    def _fallback(
        candidates: List[MemoryCandidate],
    ) -> List[MemoryCandidate]:

        """
        LLM Judge 失败时的保守降级。

        不建议全部保存，否则 LLM 故障时会污染长期记忆。
        """

        accepted = []

        for candidate in candidates:

            importance = float(
                candidate.importance or 0.0
            )

            if importance >= 0.7:
                accepted.append(candidate)

        logger.warning(
            "[MemoryJudge] 使用 fallback，"
            f"原候选={len(candidates)} "
            f"保留={len(accepted)}"
        )

        return accepted
