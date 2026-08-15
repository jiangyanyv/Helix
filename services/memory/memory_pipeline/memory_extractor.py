from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
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
    你是“长期记忆候选信息抽取器”。

    任务：从【用户消息】中发现可能值得长期保存的记忆候选。
    【AI回复】只能用于辅助理解上下文、指代和人物关系，不能作为新的事实来源。

    你只负责发现候选，不判断最终是否保存。最终是否保存由 MemoryJudge 决定。

    ====================
    一、核心原则
    ====================

    1. 记忆事实必须来自用户消息。
    2. AI回复只能辅助理解，不能将其中的推测、评价、建议、玩笑或编造内容当成用户事实。
    3. 不允许猜测，不允许为了凑数量生成记忆。
    4. 同一事实只生成一个候选。
    5. 没有明确候选时返回：
       {"memories": []}
    6. 普通寒暄、简单问答和无意义闲聊不要提取。

    ====================
    二、记忆类型
    ====================

    1. profile
    关于用户本人长期或相对稳定的信息。

    适合：
    - 身份、职业
    - 学习方向、长期目标
    - 兴趣爱好
    - 稳定偏好
    - 沟通偏好
    - 长期习惯
    - 稳定背景

    只提取本次对话中新发现的属性，使用 profile.patch，不重复整个画像。

    例如：
    用户：“我是一名Java后端开发，最近准备转AI Agent。”
    提取：
    {
      "career": "Java后端开发",
      "career_goal": "转向AI Agent开发"
    }

    不要提取明显临时信息：
    “我今天想吃火锅。”

    --------------------

    2. relationship
    表示用户与其他现实人物之间的明确关系。

    只有能确认该人物与用户存在现实关系时才提取。

    例如：
    “老王是我大学时期认识的朋友。”
    → canonical_name="老王", relation="朋友"

    “王哥就是老王，他现在在北京做软件工程师。”
    → canonical_name="老王"
    → aliases=["王哥"]
    → relation="朋友"
    → extra={"location":"北京","occupation":"软件工程师"}

    如果能确认存在关系但无法确定具体关系：
    relation="unknown"

    人物仅出现在用户消息中，但无法确认与用户存在关系时，不创建 relationship。

    不要输出 person_id。
    LLM只负责提取人物名称，后续由RelationshipService解析实体。

    --------------------

    3. episodic
    表示用户明确讲述的具体经历、事件、决定或未来可能具有回忆价值的信息。

    适合：
    - 重要经历
    - 游戏成就
    - 项目经历
    - 特殊事件
    - 重要决定
    - 重要计划
    - 与重要人物发生的事件

    例如：
    “我昨天终于打上超凡大师了。”
    → content="用户在英雄联盟排位中成功晋级超凡大师。"
    → tags=["英雄联盟","排位","超凡大师"]

    如果涉及人物：
    person_names=["老王"]

    person_names只能填写用户消息中明确出现的人物名称，不得填写person_id。

    emotion必须是对象或null：
    {
      "type": "伤感",
      "intensity": 0.8,
      "subject": "user"
    }

    没有明确情绪证据时使用null。

    timestamp只有能够从用户消息明确判断事件时间时填写，否则null。

    source固定为"conversation"。

    ====================
    三、字段规则
    ====================

    importance：
    候选记忆的重要程度初步估计，范围0.0~1.0。
    仅供MemoryJudge参考，不代表最终保存决定。

    profile：
    {
      "patch": {},
      "replace": false
    }

    只填写本次新发现的用户属性。

    relationship：
    {
      "canonical_name": "...",
      "aliases": [],
      "relation": "...",
      "confidence": 0.0,
      "extra": {}
    }

    episodic：
    {
      "content": "...",
      "tags": [],
      "person_names": [],
      "emotion": null,
      "timestamp": null,
      "source": "conversation"
    }

    ====================
    四、输出格式
    ====================

    只能输出合法JSON，不得输出Markdown、解释文字或其他内容。

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

    memory_type只能是：
    "profile"
    "relationship"
    "episodic"

    未使用的metadata字段必须为null。
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
    # Public: 批量提取（多轮消息一次抽取）
    # =========================================================

    def extract_from_messages(
        self,
        messages: List[BaseMessage],
    ) -> List[MemoryCandidate]:
        """从一批多轮对话消息中批量提取记忆候选。

        与 extract() 的区别：
            - extract() 面向单轮（user_text + ai_text）
            - extract_from_messages() 面向多轮（messages 列表）

        一次 LLM 调用产出这批消息里的所有候选，
        有多个独立事件就输出多个，没有则返回空列表。
        """

        if not messages:
            return []

        prompt = self._build_messages_prompt(messages)

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
                "[MemoryExtractor] 批量抽取完成 | "
                f"msgs={len(messages)} "
                f"candidates={len(candidates)}"
            )

            return candidates

        except Exception as e:
            logger.exception(
                f"[MemoryExtractor] 批量抽取失败: {e}"
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

    @staticmethod
    def _build_messages_prompt(
        messages: List[BaseMessage],
    ) -> str:
        """构建多轮对话批量提取 prompt。

        将整批消息按轮次编号拼入 prompt，
        要求 LLM 通读全部后一次性输出所有候选。
        """

        lines: List[str] = []

        for index, msg in enumerate(messages, 1):

            if isinstance(msg, HumanMessage):
                role = "用户"
            elif isinstance(msg, AIMessage):
                role = "AI"
            else:
                role = msg.__class__.__name__

            content = (
                msg.content
                if isinstance(msg.content, str)
                else str(msg.content)
            )

            lines.append(
                f"[{index}] {role}: {content}"
            )

        dialogue_text = "\n".join(lines)

        return f"""
请分析下面这批多轮对话（共 {len(messages)} 条消息）。

通读全部对话后，从中提取所有值得长期保存的记忆候选：
- 这批对话里可能包含多个独立事件，每个独立事件都应单独提取为一个候选；
- 同一个事实不要重复提取；
- 没有值得提取的候选时返回空数组。

【对话记录】
----------------
{dialogue_text}

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