"""ConversationManager：会话历史管理器。

职责：

1. 保存聊天历史
   - add_user_message()
   - add_ai_message()

2. 获取聊天历史
   - get_messages()
   - 返回：摘要 + 最近 M 轮原文

3. 控制上下文长度
   - 滑动窗口（MAX_HISTORY_TURNS）
   - 滚动摘要（Summarizer）

4. Redis 存储结构
   - conversation:{user_id}:messages    List    消息列表（每条带 seq、ts）
   - conversation:{user_id}:seq         String  消息序号计数器
   - conversation:{user_id}:summary     String  滚动摘要文本
   - conversation:{user_id}:{sid}:state Hash    状态（summary_cursor 等）

   messages 单条消息结构：{"seq": N, "t": "H"/"A", "c": content, "ts": unix_timestamp}
   - seq: 全局递增消息序号
   - t: 消息类型（H=用户，A=AI）
   - c: 消息内容
   - ts: Unix 时间戳（秒级，消息写入时刻）

   messages 与 seq 共享 TTL。
   summary 与 state 共享 TTL。

注意：
- Redis 作为唯一的会话历史存储
- Redis 不可用时直接抛出异常
- seq 是全局递增的消息序号，用于摘要边界追踪
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from loguru import logger

from config import Config
from core.session.summarizer import Summarizer
from infrastructure.redis.redis_client import get_redis


# ============================================================
# 常量
# ============================================================

DEFAULT_SESSION_ID = "main"

# state Hash 字段名
FIELD_SUMMARY_CURSOR = "summary_cursor"


# ============================================================
# 消息序列化
# ============================================================


def _msg_to_dict(
    message: BaseMessage,
    seq: int,
) -> dict:
    """LangChain Message -> Redis JSON 数据（带 seq 和时间戳）。"""

    if isinstance(message, HumanMessage):

        msg_type = "H"

    elif isinstance(message, AIMessage):

        msg_type = "A"

    else:

        msg_type = message.__class__.__name__

    return {
        "seq": seq,
        "t": msg_type,
        "c": message.content,
        "ts": int(time.time()),
    }


def _dict_to_msg(data: dict) -> Optional[BaseMessage]:
    """Redis JSON 数据 -> LangChain Message。

    将消息中的 ts（Unix 秒级时间戳）转换为
    'yyyy-MM-dd HH:mm:ss' 格式，存入 additional_kwargs['timestamp']。
    """

    if not isinstance(data, dict):

        return None

    message_type = data.get("t")

    content = data.get("c", "")

    # 提取时间戳并格式化
    time_str = _format_time(data.get("ts"))

    kwargs = {
        "time": time_str,
    }

    if message_type == "H":

        return HumanMessage(
            content=content,
            additional_kwargs=kwargs,
        )

    if message_type == "A":

        return AIMessage(
            content=content,
            additional_kwargs=kwargs,
        )

    # 未知类型暂时忽略
    logger.warning(
        f"[ConvMgr] 未知消息类型，跳过: {message_type}"
    )

    return None


def _format_time(ts) -> str:
    """将 Unix 秒级时间戳格式化为 'yyyy-MM-dd HH:mm:ss'。

    无时间戳或格式异常时返回空字符串。
    """

    if not ts:

        return ""

    try:

        return datetime.fromtimestamp(
            int(ts),
        ).strftime("%Y-%m-%d %H:%M:%S")

    except (TypeError, ValueError, OSError) as e:

        logger.warning(
            f"[ConvMgr] 时间戳格式化失败，ts={ts}, error={e}"
        )

        return ""


def _dict_seq(data: dict) -> int:
    """从消息 dict 中提取 seq。"""

    try:

        return int(data.get("seq", 0))

    except (TypeError, ValueError):

        return 0


# ============================================================
# Redis Key 生成
# ============================================================


def _key_prefix(user_id: str) -> str:
    """消息相关 key 的公共前缀。"""

    return (
        f"{Config.REDIS_KEY_PREFIX}:"
        f"conversation:{user_id}"
    )


def _key_messages(user_id: str) -> str:
    """消息列表 key。"""

    return f"{_key_prefix(user_id)}:messages"


def _key_seq(user_id: str) -> str:
    """消息序号计数器 key。"""

    return f"{_key_prefix(user_id)}:seq"


def _key_summary(user_id: str) -> str:
    """摘要文本 key。"""

    return f"{_key_prefix(user_id)}:summary"


def _key_state(
    user_id: str,
    session_id: str,
) -> str:
    """会话状态 key。"""

    return (
        f"{_key_prefix(user_id)}:"
        f"{session_id}:state"
    )


# ============================================================
# ConversationManager
# ============================================================


class ConversationManager:
    """会话历史管理器。

    Redis 存储结构：

        conversation:{user_id}:messages
            └── List[{"seq": N, "t": "H"/"A", "c": content, "ts": unix_timestamp}]
                - seq: 全局递增消息序号
                - t: 消息类型（H=用户，A=AI）
                - c: 消息内容
                - ts: Unix 时间戳（秒级，消息写入时刻）

        conversation:{user_id}:seq
            └── Int（INCR 递增，作为消息唯一序号）

        conversation:{user_id}:summary
            └── String（滚动摘要文本）

        conversation:{user_id}:{session_id}:state
            └── Hash
                ├── summary_cursor: Int（已摘要到的 seq）

    读取时返回：
        summary（如有）+ 最近 M 轮消息
    """

    def __init__(
        self,
        summarizer: Optional[Summarizer] = None,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> None:

        # Summarizer 用于滚动摘要（由 Container 注入）
        self._summarizer = summarizer

        # session_id 用于 state Hash 隔离
        # 单用户单机部署默认 "main"
        self._session_id = session_id

        # Redis 客户端懒加载
        self._redis = None

    # ========================================================
    # Internal: Redis
    # ========================================================

    def _get_redis(self):
        """获取 Redis 客户端。"""

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

        返回结构：
            [SystemMessage(摘要)] + 最近 M 轮消息

        如果有摘要：
            - 摘要作为 SystemMessage 前置
            - 只返回 seq > summary_cursor 的消息
            - 再应用滑动窗口

        如果没有摘要：
            - 返回全部消息（应用滑动窗口）
        """

        redis_client = self._get_redis()

        msg_key = _key_messages(user_id)

        try:

            raw_list = redis_client.lrange(
                msg_key,
                0,
                -1,
            )

            # 读取 summary_cursor
            cursor = self._get_summary_cursor(user_id)

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

                # 只取 cursor 之后的消息
                if _dict_seq(data) <= cursor:

                    continue

                message = _dict_to_msg(data)

                if message is not None:

                    messages.append(message)

            # 应用滑动窗口
            messages = self._apply_window(
                user_id,
                messages,
            )

            # 如果有摘要，前置 SystemMessage
            summary = self._get_summary(user_id)

            if summary:

                summary_msg = SystemMessage(
                    content=(
                        "【历史对话摘要】\n"
                        "----------------\n"
                        f"{summary}"
                    )
                )

                return [summary_msg] + messages

            return messages

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
        """清空指定用户的会话历史。

        清除：messages / seq / summary / state
        """

        redis_client = self._get_redis()

        keys = [
            _key_messages(user_id),
            _key_seq(user_id),
            _key_summary(user_id),
            _key_state(user_id, self._session_id),
        ]

        try:

            redis_client.delete(*keys)

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
    # Internal: 写入消息
    # ========================================================

    def _append_one(
        self,
        user_id: str,
        message: BaseMessage,
    ) -> None:
        """向 Redis 追加一条消息。

        流程：
            1. INCR seq → 得到新 seq
            2. RPUSH messages {seq, t, c}
            3. 刷新 messages + seq 的 TTL（共享）
            4. 检查摘要触发
        """

        redis_client = self._get_redis()

        seq_key = _key_seq(user_id)

        msg_key = _key_messages(user_id)

        try:

            # 1. INCR seq
            seq = redis_client.incr(seq_key)

            # 2. 构造消息
            payload = json.dumps(
                _msg_to_dict(message, seq),
                ensure_ascii=False,
            )

            redis_client.rpush(
                msg_key,
                payload,
            )

            # 3. 四组 key 共享 TTL（同步刷新，避免部分过期）
            self._refresh_all_ttl(user_id)

        except Exception as e:  # noqa: BLE001

            logger.error(
                f"[ConvMgr] Redis 保存消息失败，"
                f"user_id={user_id}, error={e}"
            )

            raise

        # # 4. 摘要触发检查（失败不影响消息保存）
        # self._check_summary_trigger(user_id)
            # 修改：只在AI消息后触发摘要
        if isinstance(message, AIMessage):
            self._check_summary_trigger(user_id)  # ← 只在AI回复后触发



    # ========================================================
    # Internal: 滑动窗口
    # ========================================================

    def _apply_window(
        self,
        user_id: str,
        messages: List[BaseMessage],
    ) -> List[BaseMessage]:
        """应用滑动窗口，保留最近 MAX_HISTORY_TURNS 轮消息。"""

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

    # ========================================================
    # Internal: 摘要触发
    # ========================================================

    def _check_summary_trigger(
        self,
        user_id: str,
    ) -> None:
        """检查是否达到摘要触发阈值。

        触发条件：
            未摘要的消息轮次 > 2 * SUMMARY_TRIGGER_THRESHOLD

        触发后：
            - 保留最近 SUMMARY_TRIGGER_THRESHOLD 轮不摘要
            - 其余部分与旧摘要合并，生成新摘要
            - summary_cursor 推进到 max_seq - 2 * keep_turns
        """

        if self._summarizer is None:

            # Summarizer 未注入，跳过（开发环境降级）
            return

        try:

            redis_client = self._get_redis()

            # 当前最大 seq
            max_seq = int(
                redis_client.get(_key_seq(user_id)) or 0
            )

            # 已摘要到的 seq
            cursor = self._get_summary_cursor(user_id)

            # 未摘要的消息数
            unsummarized_count = max_seq - cursor

            # 未摘要的轮次（1 轮 = 1 H + 1 A = 2 条）
            unsummarized_turns = unsummarized_count // 2

            keep_turns = Config.SUMMARY_TRIGGER_THRESHOLD

            # 触发阈值 = 2 * keep_turns
            # （需要积累足够多的未摘要消息才触发，
            #   避免每轮都触发摘要）
            trigger_threshold = keep_turns * 2

            if unsummarized_turns < trigger_threshold:

                return

            # 保留最近 keep_turns 轮（2 * keep_turns 条消息）
            keep_count = keep_turns * 2

            summarize_to = max_seq - keep_count

            if summarize_to <= cursor:

                # 没有新消息需要摘要
                return

            # 读取需要摘要的消息（cursor < seq <= summarize_to）
            to_summarize = self._read_messages_in_range(
                user_id,
                start_seq=cursor + 1,
                end_seq=summarize_to,
            )

            if not to_summarize:

                return

            # 读取旧摘要
            old_summary = self._get_summary(user_id)

            logger.info(
                f"[ConvMgr] 触发摘要 | "
                f"user_id={user_id} | "
                f"max_seq={max_seq} | "
                f"cursor={cursor} | "
                f"summarize_to={summarize_to} | "
                f"msgs={len(to_summarize)}"
            )

            # 调用 Summarizer
            new_summary = self._summarizer.summarize(
                messages=to_summarize,
                old_summary=old_summary,
            )

            if new_summary is None:

                logger.warning(
                    "[ConvMgr] 摘要生成失败，"
                    "cursor 不推进"
                )

                return

            # 存储新摘要
            self._set_summary(user_id, new_summary)

            # 更新 cursor
            self._set_summary_cursor(user_id, summarize_to)

            logger.info(
                f"[ConvMgr] 摘要完成 | "
                f"user_id={user_id} | "
                f"new_cursor={summarize_to} | "
                f"summary_len={len(new_summary)}"
            )

        except Exception as e:  # noqa: BLE001

            # 摘要失败不影响消息保存
            logger.warning(
                f"[ConvMgr] 摘要触发检查失败: {e}"
            )

    # ========================================================
    # Internal: 读取指定 seq 范围的消息
    # ========================================================

    def _read_messages_in_range(
        self,
        user_id: str,
        start_seq: int,
        end_seq: int,
    ) -> List[BaseMessage]:
        """读取 seq 在 [start_seq, end_seq] 范围内的消息。"""

        redis_client = self._get_redis()

        msg_key = _key_messages(user_id)

        raw_list = redis_client.lrange(
            msg_key,
            0,
            -1,
        )

        result: List[BaseMessage] = []

        for raw in raw_list:

            try:

                data = (
                    json.loads(raw)
                    if isinstance(raw, str)
                    else raw
                )

            except Exception:  # noqa: BLE001

                continue

            seq = _dict_seq(data)

            if start_seq <= seq <= end_seq:

                message = _dict_to_msg(data)

                if message is not None:

                    result.append(message)

        return result

    # ========================================================
    # Internal: 四组 key TTL 同步刷新
    # ========================================================

    def _refresh_all_ttl(
        self,
        user_id: str,
    ) -> None:
        """同步刷新同一 user_id 的四组 key TTL。

        保证四个 key 在同一时刻过期：
          - conversation:{user_id}:messages
          - conversation:{user_id}:seq
          - conversation:{user_id}:summary
          - conversation:{user_id}:{session_id}:state

        如果某组 key 不存在（尚未写入 summary/state），
        redis.expire 会返回 False，安全忽略。
        """

        redis_client = self._get_redis()

        ttl = Config.HISTORY_CACHE_TTL_SEC

        keys = [
            _key_messages(user_id),
            _key_seq(user_id),
            _key_summary(user_id),
            _key_state(user_id, self._session_id),
        ]

        for key in keys:

            try:

                redis_client.expire(key, ttl)

            except Exception as e:  # noqa: BLE001

                logger.warning(
                    f"[ConvMgr] 刷新TTL失败，"
                    f"user_id={user_id}, key={key}, error={e}"
                )

    # ========================================================
    # Internal: Summary 读写
    # ========================================================

    def _get_summary(
        self,
        user_id: str,
    ) -> Optional[str]:
        """读取摘要文本。"""

        redis_client = self._get_redis()

        try:

            summary = redis_client.get(
                _key_summary(user_id)
            )

            return summary if summary else None

        except Exception as e:  # noqa: BLE001

            logger.warning(
                f"[ConvMgr] 读取摘要失败: {e}"
            )

            return None

    def _set_summary(
        self,
        user_id: str,
        summary: str,
    ) -> None:
        """写入摘要文本，并同步刷新四组 key TTL。"""

        redis_client = self._get_redis()

        summary_key = _key_summary(user_id)

        try:

            redis_client.set(
                summary_key,
                summary,
            )

            # 摘要写入后同步四组 key TTL
            self._refresh_all_ttl(user_id)

        except Exception as e:  # noqa: BLE001

            logger.error(
                f"[ConvMgr] 保存摘要失败: {e}"
            )

            raise

    # ========================================================
    # Internal: State 读写（summary_cursor）
    # ========================================================

    def _get_summary_cursor(
        self,
        user_id: str,
    ) -> int:
        """读取 summary_cursor（已摘要到的 seq）。

        默认 0（未摘要）。
        """

        redis_client = self._get_redis()

        try:

            value = redis_client.hget(
                _key_state(user_id, self._session_id),
                FIELD_SUMMARY_CURSOR,
            )

            if value is None:

                return 0

            return int(value)

        except Exception as e:  # noqa: BLE001

            logger.warning(
                f"[ConvMgr] 读取 summary_cursor 失败: {e}"
            )

            return 0

    def _set_summary_cursor(
        self,
        user_id: str,
        cursor: int,
    ) -> None:
        """写入 summary_cursor，并同步刷新四组 key TTL。"""

        redis_client = self._get_redis()

        state_key = _key_state(user_id, self._session_id)

        try:

            redis_client.hset(
                state_key,
                FIELD_SUMMARY_CURSOR,
                int(cursor),
            )

            # cursor 写入后同步四组 key TTL
            self._refresh_all_ttl(user_id)

        except Exception as e:  # noqa: BLE001

            logger.error(
                f"[ConvMgr] 保存 summary_cursor 失败: {e}"
            )

            raise
