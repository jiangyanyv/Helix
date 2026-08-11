"""ConversationManager：会话历史管理器。

职责：

1. 保存聊天历史
   - add_user_message()
   - add_ai_message()

2. 获取聊天历史
   - get_messages()

3. 控制上下文长度
   - MAX_HISTORY_TURNS：硬上限（滑动窗口）
   - SUMMARY_TRIGGER_THRESHOLD：软摘要触发

4. Redis：
   - Redis 作为唯一的会话历史存储
   - 每次写入刷新 TTL
   - Redis 连接由 infrastructure.redis.client 负责

注意：
- 不再使用进程内内存作为 fallback。
- Redis 不可用时直接抛出异常，由上层决定如何处理。
"""

from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from loguru import logger

from config import Config
from infrastructure.redis.redis_client import get_redis


# ============================================================
# 消息序列化
# ============================================================


def _msg_to_dict(message: BaseMessage) -> dict:
    """LangChain Message -> Redis JSON 数据。"""

    if isinstance(message, HumanMessage):
        return {
            "t": "H",
            "c": message.content,
        }

    if isinstance(message, AIMessage):
        return {
            "t": "A",
            "c": message.content,
        }

    # 兜底：记录实际消息类型
    return {
        "t": message.__class__.__name__,
        "c": message.content,
    }


def _dict_to_msg(data: dict) -> Optional[BaseMessage]:
    """Redis JSON 数据 -> LangChain Message。"""

    if not isinstance(data, dict):
        return None

    message_type = data.get("t")
    content = data.get("c", "")

    if message_type == "H":
        return HumanMessage(content=content)

    if message_type == "A":
        return AIMessage(content=content)

    # 目前只支持 HumanMessage / AIMessage。
    # 未知类型暂时忽略，避免错误数据导致整个会话读取失败。
    logger.warning(
        f"[ConvMgr] 未知消息类型，跳过: {message_type}"
    )

    return None


# ============================================================
# Redis Key
# ============================================================


def _redis_key(user_id: str) -> str:
    """生成用户会话历史 Redis Key。"""

    return f"{Config.REDIS_KEY_PREFIX}:conv:{user_id}"


# ============================================================
# ConversationManager
# ============================================================


class ConversationManager:
    """会话历史管理器。

    Redis 是唯一的会话历史存储。

    Redis：
        └── conv:{user_id}
                ├── HumanMessage
                ├── AIMessage
                ├── HumanMessage
                └── AIMessage

    ConversationManager 只负责：
        - 会话消息读写
        - Redis Key
        - TTL
        - 滑动窗口
        - 摘要触发
    """

    def __init__(self):
        # Redis 客户端懒加载
        #
        # None 表示尚未获取 Redis 客户端
        self._redis = None

    # ========================================================
    # Internal
    # ========================================================

    def _get_redis(self):
        """获取 Redis 客户端。

        Redis 客户端本身由 infrastructure.redis.client 管理。
        """

        if self._redis is None:
            self._redis = get_redis()

        if self._redis is None:
            raise RuntimeError(
                "Redis 不可用，ConversationManager 无法工作"
            )

        return self._redis

    # ========================================================
    # Public API
    # ========================================================

    def get_messages(
        self,
        user_id: str,
    ) -> List[BaseMessage]:
        """获取历史消息。

        自动应用 MAX_HISTORY_TURNS 滑动窗口。

        Returns:
            最近的历史消息列表。
        """

        redis_client = self._get_redis()

        key = _redis_key(user_id)

        try:
            raw_list = redis_client.lrange(
                key,
                0,
                -1,
            )

            messages: List[BaseMessage] = []

            for raw in raw_list:
                try:
                    data = (
                        json.loads(raw)
                        if isinstance(raw, str)
                        else raw
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"[ConvMgr] Redis 消息 JSON 解析失败，"
                        f"user_id={user_id}, error={e}"
                    )
                    continue

                message = _dict_to_msg(data)

                if message is not None:
                    messages.append(message)

            return self._apply_window(
                user_id,
                messages,
            )

        except Exception as e:  # noqa: BLE001
            logger.error(
                f"[ConvMgr] Redis 获取会话失败，"
                f"user_id={user_id}, error={e}"
            )
            raise

    # ========================================================

    def add_user_message(
        self,
        user_id: str,
        content: str,
    ) -> None:
        """添加用户消息。"""

        self._append_one(
            user_id,
            HumanMessage(content=content),
        )

    # ========================================================

    def add_ai_message(
        self,
        user_id: str,
        content: str,
    ) -> None:
        """添加 AI 消息。"""

        self._append_one(
            user_id,
            AIMessage(content=content),
        )

    # ========================================================

    def clear(
        self,
        user_id: str,
    ) -> None:
        """清空指定用户的会话历史。"""

        redis_client = self._get_redis()

        key = _redis_key(user_id)

        try:
            redis_client.delete(key)

            logger.info(
                f"[ConvMgr] 清空会话 user_id={user_id}"
            )

        except Exception as e:  # noqa: BLE001
            logger.error(
                f"[ConvMgr] Redis 删除会话失败，"
                f"user_id={user_id}, error={e}"
            )
            raise

    # ========================================================
    # Internal helpers
    # ========================================================

    def _append_one(
        self,
        user_id: str,
        message: BaseMessage,
    ) -> None:
        """向 Redis 追加一条消息。"""

        redis_client = self._get_redis()

        key = _redis_key(user_id)

        payload = json.dumps(
            _msg_to_dict(message),
            ensure_ascii=False,
        )

        try:
            # 添加消息
            redis_client.rpush(
                key,
                payload,
            )

            # 每次写入刷新 TTL
            redis_client.expire(
                key,
                Config.HISTORY_CACHE_TTL_SEC,
            )

        except Exception as e:  # noqa: BLE001
            logger.error(
                f"[ConvMgr] Redis 保存消息失败，"
                f"user_id={user_id}, error={e}"
            )
            raise

        # 软摘要触发
        self._check_summary_trigger(user_id)

    # ========================================================

    def _check_summary_trigger(
        self,
        user_id: str,
    ) -> None:
        """检查是否达到摘要触发阈值。

        当前只记录日志。
        后续可以在这里接入 Summarizer。
        """

        try:
            current_count = self._raw_count(
                user_id
            )

            current_turns = current_count // 2

            if (
                current_turns
                > Config.SUMMARY_TRIGGER_THRESHOLD
                > 0
            ):
                logger.debug(
                    f"[ConvMgr] 会话轮次达到摘要阈值: "
                    f"user_id={user_id}, "
                    f"turns={current_turns}, "
                    f"threshold="
                    f"{Config.SUMMARY_TRIGGER_THRESHOLD}"
                    f"（summarizer 待实现，"
                    f"当前仍依赖滑动窗口裁剪）"
                )

        except Exception as e:  # noqa: BLE001
            # 摘要触发检查失败不影响消息保存
            logger.warning(
                f"[ConvMgr] 摘要阈值检查失败: {e}"
            )

    # ========================================================

    def _raw_count(
        self,
        user_id: str,
    ) -> int:
        """获取 Redis 中当前会话的实际消息数量。"""

        redis_client = self._get_redis()

        try:
            count = redis_client.llen(
                _redis_key(user_id)
            )

            return int(count) if count else 0

        except Exception as e:  # noqa: BLE001
            logger.error(
                f"[ConvMgr] Redis 获取消息数量失败，"
                f"user_id={user_id}, error={e}"
            )
            raise

    # ========================================================

    def _apply_window(
        self,
        user_id: str,
        messages: List[BaseMessage],
    ) -> List[BaseMessage]:
        """应用滑动窗口。

        保留最近 MAX_HISTORY_TURNS 轮消息。

        例如：

            MAX_HISTORY_TURNS = 5

        最多返回：

            Human
            AI
            Human
            AI
            Human
            AI
            Human
            AI
            Human
            AI
        """

        max_msgs = max(
            2,
            int(Config.MAX_HISTORY_TURNS) * 2,
        )

        if len(messages) <= max_msgs:
            return messages

        trimmed = messages[-max_msgs:]

        logger.debug(
            f"[ConvMgr] 滑动窗口裁剪 "
            f"user_id={user_id} "
            f"from={len(messages)} "
            f"to={len(trimmed)} "
            f"(max_turns="
            f"{Config.MAX_HISTORY_TURNS})"
        )

        return trimmed
