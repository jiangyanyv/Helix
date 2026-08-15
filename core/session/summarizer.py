"""Summarizer：对话滚动摘要器。

职责：

    将超过阈值的对话历史压缩成摘要，
    形成「摘要 + 最近 M 轮原文」的混合上下文，
    避免长对话中早期信息被滑动窗口丢弃。

轮转机制：

    第 1 次触发：
        旧摘要 = None
        新增待摘要消息 = [seq 1 .. seq K]
        新摘要 = summarize(None, 新增消息)

    第 2 次触发：
        旧摘要 = 第 1 次的摘要
        新增待摘要消息 = [seq K+1 .. seq L]
        新摘要 = summarize(旧摘要, 新增消息)

    每次 summarize 都把旧摘要 + 新消息合并成完整新摘要，
    summary_cursor 推进到 L，下次从 L+1 开始算。

不负责：
    - 判断何时触发（由 ConversationManager 决定）
    - 存储 summary / cursor（由 ConversationManager 负责）
    - 读取历史消息（由 ConversationManager 负责）
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from loguru import logger

from services.llm.chat_request import ChatRequest
from services.llm.deepseek_client import DeepSeekClient
from core.session.summarizer_system_prompt import SUMMARIZER_SYSTEM_PROMPT

from config import Config


class Summarizer:
    """对话滚动摘要器。

    使用 LLM 将对话历史压缩成摘要。
    支持轮转：旧摘要 + 新消息 → 新摘要。
    """

    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
    ) -> None:

        self.llm = llm_client or DeepSeekClient()

    # ========================================================
    # Public
    # ========================================================

    def summarize(
        self,
        messages: List[BaseMessage],
        old_summary: Optional[str] = None,
    ) -> Optional[str]:
        """生成滚动摘要。

        Args:
            messages: 本次需要摘要的新增对话消息。
            old_summary: 上一次的摘要文本（如有）。

        Returns:
            更新后的完整摘要文本。失败时返回 old_summary（降级）。
        """

        if not messages:
            return old_summary

        try:

            # ====================================================
            # 打印待摘要的每条消息
            # ====================================================
            # logger.info(f"[Summarizer] 开始生成摘要，共 {len(messages)} 条消息:")
            # for i, msg in enumerate(messages, 1):
            #     role = (
            #         "用户" if isinstance(msg, HumanMessage)
            #         else "AI" if isinstance(msg, AIMessage)
            #         else msg.__class__.__name__
            #     )
            #     content = msg.content if isinstance(msg.content, str) else str(msg.content)
            #     # 截断过长内容，避免日志刷屏（可选）
            #     logger.info(f"  #{i} [{role}]: {content[:200]}{'...' if len(content) > 200 else ''}")
            # ====================================================

            prompt = self._build_prompt(
                messages=messages,
                old_summary=old_summary,
            )

            request = ChatRequest(
                messages=[
                    SystemMessage(
                        content=SUMMARIZER_SYSTEM_PROMPT
                    ),
                    HumanMessage(
                        content=prompt
                    ),
                ],
                model=self.llm.model,
                stream=False,
                temperature=0.0,
                top_p=1.0,
                max_tokens=Config.SUMMARY_MAX_TOKENS,
            )

            response = self.llm.generate(request)

            new_summary = (response.text or "").strip()

            if not new_summary:

                logger.warning(
                    "[Summarizer] LLM 返回空摘要，"
                    "保留旧摘要降级"
                )

                return old_summary

            logger.info(
                f"[Summarizer] 摘要生成成功 | "
                f"old_len={len(old_summary) if old_summary else 0} | "
                f"new_len={len(new_summary)} | "
                f"msg_count={len(messages)}"
            )

            return new_summary

        except Exception as e:  # noqa: BLE001

            logger.exception(
                f"[Summarizer] 摘要生成失败，"
                f"保留旧摘要降级: {e}"
            )

            return old_summary

    # ========================================================
    # Prompt 构建
    # ========================================================

    @staticmethod
    def _build_prompt(
        messages: List[BaseMessage],
        old_summary: Optional[str],
    ) -> str:

        # 格式化对话消息
        dialogue_lines: List[str] = []

        for msg in messages:

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

            dialogue_lines.append(
                f"{role}: {content}"
            )

        dialogue_text = "\n".join(dialogue_lines)

        # 拼接旧摘要
        if old_summary:

            old_summary_block = (
                "【已有摘要】\n"
                "----------------\n"
                f"{old_summary}"
            )

        else:

            old_summary_block = "【已有摘要】\n（无）"

        return f"""
请根据以下内容生成更新后的完整摘要。

{old_summary_block}

【新增对话】
----------------
{dialogue_text}

请输出合并后的完整摘要（包含已有摘要的关键信息 + 新增对话的关键信息）。
""".strip()
