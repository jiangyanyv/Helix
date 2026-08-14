from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from loguru import logger

from services.llm.chat_request import ChatRequest
from services.llm.deepseek_client import DeepSeekClient

from services.memory.memory_candidate import (
    MemoryCandidate,
    MemoryCandidateSchema,
    MemoryExtractionResult,
    MemoryType,
)


class MemoryExtractor:
    """
    从用户与 AI 的对话中发现潜在长期记忆。

    职责：

        对话
          ↓
        LLM
          ↓
        JSON
          ↓
        Pydantic Schema
          ↓
        MemoryCandidate

    不负责：

        - 判断是否值得保存
        - 写数据库
        - 修改已有记忆
        - 解析 person_id
    """

    SYSTEM_PROMPT = """
你是一个“长期记忆候选信息抽取器”。

你的任务是：
从【用户消息】和【AI回复】中，发现可能值得长期保存的信息。

你只负责“发现候选记忆”，不要判断最终是否值得保存。
最终是否保存由后续 MemoryJudge 决定。

====================
一、记忆类型
====================

1. profile

表示关于“用户本人”的长期或相对稳定信息。

适合：

- 身份
- 职业
- 学习方向
- 长期目标
- 兴趣爱好
- 稳定偏好
- 沟通偏好
- 长期习惯
- 稳定背景信息

例如：

“我是一名 Java 后端开发，最近准备转 AI Agent。”

可以提取：

{
    "career": "Java后端开发",
    "career_goal": "转向AI Agent开发"
}

不要提取明显临时的信息：

“我今天想吃火锅。”

--------------------

2. relationship

表示“用户与其他现实人物之间的关系”。

只有在能够确认该人物与用户存在实际关系时才提取。

例如：

“老王是我大学时期认识的朋友。”

提取：

canonical_name = "老王"
relation = "朋友"

例如：

“王哥就是老王，他现在在北京做软件工程师。”

提取：

canonical_name = "老王"
aliases = ["王哥"]
relation = "朋友"

extra = {
    "location": "北京",
    "occupation": "软件工程师"
}

注意：

仅仅出现一个人物名字，
不代表这个人一定是用户认识的人。

例如：

“网上有个老王特别搞笑。”

不要创建 relationship。

如果无法确定具体关系，
但能够确定人物与用户存在关系，
可以使用：

relation = "unknown"

重要：

不要输出 person_id。

person_id 属于数据库实体层，
LLM 不知道数据库中的 person_id。

如果出现：

“老王”
“王哥”
“王叔”

只需要提取这些名称。

后续系统会通过 RelationshipService
自动解析它们属于哪个已有人物。

--------------------

3. episodic

表示具体发生过的事件，
或者未来聊天可能有回忆价值的信息。

适合：

- 重要经历
- 游戏成就
- 项目经历
- 特殊事件
- 用户做出的重要决定
- 重要计划
- 与重要人物发生的事件
- 对未来聊天可能有价值的经历

例如：

“我昨天终于打上超凡大师了。”

提取：

content =
“用户在英雄联盟排位中成功晋级超凡大师。”

tags =
["英雄联盟", "排位", "超凡大师"]

如果事件涉及人物：

例如：

“老王最近失联了，我很想念他。”

应该提取：

person_names = ["老王"]

注意：

person_names 是人物名称，
不是数据库 person_id。

--------------------
二、严格规则
====================

1. 用户消息是事实的主要来源。

2. AI回复只能用于理解上下文。

3. 不能把 AI 自己的推测、
评价、玩笑或编造内容当成用户事实。

4. 不允许猜测。

5. 不允许为了凑数量而生成记忆。

6. 没有值得提取的候选时：

{
    "memories": []
}

7. 同一个事实不要生成多个候选。

8. 一次对话可以产生多个不同类型的候选。

9. PROFILE 主要描述用户本人。

10. RELATIONSHIP 描述用户与其他人物的关系。

11. EPISODIC 描述具体发生过的事件。

12. 普通寒暄、简单问答、
无意义闲聊不要直接作为长期记忆。

13. 不要把 AI 回复中的人格化表达、
假设、建议当成用户事实。

--------------------
三、PROFILE
====================

profile.patch 只填写本次对话中新发现的用户属性。

不要重复整个用户画像。

例如：

{
    "career_goal": "转向AI Agent开发"
}

而不是：

{
    "name": "...",
    "age": "...",
    "career": "...",
    ...
}

--------------------
四、RELATIONSHIP
====================

canonical_name：

人物主要名称。

aliases：

其他明确称呼。

relation：

人物与用户之间的关系。

confidence：

0~1。

extra：

其他明确的人物信息。

不要猜测人物信息。

不要输出 person_id。

--------------------
五、EPISODIC
====================

person_names：

事件涉及的人物名称。

例如：

{
    "person_names": ["老王"]
}

如果没有明确涉及人物：

[]

注意：

这里只能输出人物名称，
不能输出数据库 person_id。

emotion：

必须是对象或者 null。

正确：

{
    "type": "伤感",
    "intensity": 0.8,
    "subject": "user"
}

禁止：

"伤感"

没有明确情绪证据时：

null

timestamp：

只有能够从对话明确判断事件时间时填写，
否则 null。

source：

固定使用 "conversation"。

--------------------
六、importance
====================

importance 表示候选记忆的重要程度初始估计。

范围：

0.0 ~ 1.0

只是初步估计，
最终是否保存由 MemoryJudge 决定。

--------------------
七、输出格式
====================

只能输出 JSON。

禁止：

- Markdown
- ```json
- 解释文字
- JSON 前后的其他文字

格式：

{
    "memories": [
        {
            "memory_type": "profile",
            "content": "...",
            "tags": [],
            "importance": 0.5,
            "profile": {
                "patch": {},
                "replace": false
            },
            "relationship": null,
            "episodic": null
        }
    ]
}

memory_type 只能是：

"profile"
"relationship"
"episodic"

未使用的 metadata 字段必须为 null。
""".strip()

    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
    ) -> None:
        self.llm = llm_client or DeepSeekClient()

    # =========================================================
    # Public
    # =========================================================

    def extract(
        self,
        user_text: str,
        ai_text: str,
    ) -> List[MemoryCandidate]:

        if not (user_text or ai_text):
            return []

        user_text = (user_text or "").strip()
        ai_text = (ai_text or "").strip()

        prompt = self._build_prompt(
            user_text=user_text,
            ai_text=ai_text,
        )

        try:
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
                    "[MemoryExtractor] LLM 返回为空"
                )
                return []

            logger.debug(
                "[MemoryExtractor] LLM raw response: "
                f"{raw[:1500]}"
            )

            result = self._parse_result(raw)

            candidates = self._convert_candidates(result)

            logger.info(
                "[MemoryExtractor] 抽取完成 | "
                f"candidates={len(candidates)}"
            )

            return candidates

        except Exception as e:
            logger.exception(
                f"[MemoryExtractor] 抽取失败: {e}"
            )
            return []

    # =========================================================
    # Prompt
    # =========================================================

    @staticmethod
    def _build_prompt(
        user_text: str,
        ai_text: str,
    ) -> str:

        return f"""
请分析下面这次对话。

只提取用户明确表达或可以直接确认的长期记忆候选。

【用户消息】
----------------
{user_text or "(空)"}

【AI回复】
----------------
{ai_text or "(空)"}

严格按照系统要求输出 JSON。
""".strip()

    # =========================================================
    # JSON Parse
    # =========================================================

    @staticmethod
    def _parse_result(
        raw: str,
    ) -> MemoryExtractionResult:

        cleaned = raw.strip()

        # 兼容模型错误输出 Markdown code fence
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

        # 直接解析
        try:
            data = json.loads(cleaned)

        except json.JSONDecodeError:

            # 尝试从文本中提取 JSON object
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start < 0 or end <= start:
                raise ValueError(
                    "LLM 返回内容中没有有效 JSON"
                )

            json_text = cleaned[
                start:end + 1
            ]

            data = json.loads(
                json_text
            )

        # Pydantic 校验
        return MemoryExtractionResult.model_validate(
            data
        )

    # =========================================================
    # Convert
    # =========================================================

    @staticmethod
    def _convert_candidates(
        result: MemoryExtractionResult,
    ) -> List[MemoryCandidate]:

        candidates: List[MemoryCandidate] = []

        for item in result.memories:

            content = (
                item.content or ""
            ).strip()

            if not content:
                continue

            # =================================================
            # PROFILE
            # =================================================

            if item.memory_type is MemoryType.PROFILE:

                if item.profile is None:
                    logger.warning(
                        "[MemoryExtractor] "
                        "PROFILE 缺少 profile 字段"
                    )
                    continue

                metadata = {
                    "patch": item.profile.patch,
                    "replace": item.profile.replace,
                }

                candidates.append(
                    MemoryCandidate(
                        memory_type=MemoryType.PROFILE,
                        content=content,
                        tags=list(item.tags),
                        importance=float(
                            item.importance
                        ),
                        metadata=metadata,
                    )
                )

            # =================================================
            # RELATIONSHIP
            # =================================================

            elif item.memory_type is MemoryType.RELATIONSHIP:

                if item.relationship is None:
                    logger.warning(
                        "[MemoryExtractor] "
                        "RELATIONSHIP 缺少 relationship 字段"
                    )
                    continue

                relationship = item.relationship

                canonical_name = (
                    relationship.canonical_name.strip()
                )

                if not canonical_name:
                    logger.warning(
                        "[MemoryExtractor] "
                        "RELATIONSHIP canonical_name为空"
                    )
                    continue

                aliases: List[str] = []

                for alias in relationship.aliases:

                    if not alias:
                        continue

                    alias = alias.strip()

                    if (
                        alias
                        and alias.lower()
                        != canonical_name.lower()
                        and alias.lower()
                        not in {
                            x.lower()
                            for x in aliases
                        }
                    ):
                        aliases.append(alias)

                metadata = {
                    "canonical_name": canonical_name,
                    "aliases": aliases,
                    "relation": (
                        relationship.relation
                        or "unknown"
                    ),
                    "confidence": float(
                        relationship.confidence
                    ),
                    "extra": dict(
                        relationship.extra
                    ),
                }

                candidates.append(
                    MemoryCandidate(
                        memory_type=MemoryType.RELATIONSHIP,
                        content=canonical_name,
                        tags=list(item.tags),
                        importance=float(
                            item.importance
                        ),
                        metadata=metadata,
                    )
                )

            # =================================================
            # EPISODIC
            # =================================================

            elif item.memory_type is MemoryType.EPISODIC:

                if item.episodic is None:
                    logger.warning(
                        "[MemoryExtractor] "
                        "EPISODIC 缺少 episodic 字段"
                    )
                    continue

                episodic = item.episodic

                # 人物名称，而不是 person_id
                person_names: List[str] = []

                for name in episodic.person_names:

                    if not name:
                        continue

                    name = name.strip()

                    if not name:
                        continue

                    if name.lower() not in {
                        x.lower()
                        for x in person_names
                    }:
                        person_names.append(name)

                metadata: Dict[str, Any] = {
                    # "person_ids": list(episodic.person_ids),
                    "person_names": list(episodic.person_names),
                    "timestamp": episodic.timestamp,
                    "source": (
                        episodic.source
                        or "conversation"
                    ),
                }

                if episodic.emotion is not None:
                    metadata["emotion"] = (
                        episodic.emotion.model_dump()
                    )

                candidates.append(
                    MemoryCandidate(
                        memory_type=MemoryType.EPISODIC,
                        content=content,
                        tags=list(item.tags),
                        importance=float(
                            item.importance
                        ),
                        metadata=metadata,
                    )
                )

        return candidates