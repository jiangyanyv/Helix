"""TTL 兜底记忆抽取扫描器。

背景：
    记忆抽取已改为"跟随摘要触发"，若一批消息长期未达摘要阈值、
    同时用户 7 天未活跃（TTL 到期），这批消息会在 Redis 中消失，
    其中若有重要内容便永久遗失。

机制：
    启动一个后台 daemon 线程：
        - 启动后立即执行第 1 次扫描
        - 之后每隔 SCAN_INTERVAL_SEC（默认 1 小时）扫描一次

    对每个 user 的 messages key：
        1. 进入兜底窗口（0 ≤ TTL < RESCUE_WINDOW_SEC，默认 1 小时）：
           把 summary_cursor 之后、尚未抽取的消息，
           备份到 fallbackmessages key（NX + 1.5h TTL）。

        2. 下次扫描若 messages 已到期（TTL == -2）：
           从同级 fallbackmessages 中取出备份消息，
           触发一次 Memory Graph 完成记忆抽取，
           之后删除 fallbackmessages。

    若用户活跃、TTL 被刷新：
        messages TTL > 1h，不再命中备份分支；
        fallbackmessages 1.5h 后自动过期，无需手动清理。

单用户单机部署下，扫描量极小，无性能压力。
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import List, Optional

from langchain_core.messages import BaseMessage
from loguru import logger

from config import Config
from core.session.conversation_manager import (
    FIELD_SUMMARY_CURSOR,
    _dict_to_msg,
    _dict_seq,
    _key_messages,
    _key_state,
    DEFAULT_SESSION_ID,
)
from infrastructure.redis.redis_client import get_redis


# ============================================================
# 常量
# ============================================================

# 扫描间隔：每 1 小时一次
SCAN_INTERVAL_SEC = 60 * 60

# 兜底窗口：messages 剩余 TTL < 1h 触发备份
RESCUE_WINDOW_SEC = 60 * 60

# fallback 备份 TTL：1.5 小时，保证覆盖至少 1 次下一轮扫描
FALLBACK_TTL_SEC = int(SCAN_INTERVAL_SEC * 1.5)

# fallback 后缀（在 user_id 同级）
FALLBACK_MESSAGES_SUFFIX = "fallbackmessages"


# ============================================================
# Key 辅助
# ============================================================

def _key_fallbackmessages(user_id: str) -> str:
    """兜底备份消息 key。"""

    return (
        f"{Config.REDIS_KEY_PREFIX}:"
        f"conversation:{user_id}:{FALLBACK_MESSAGES_SUFFIX}"
    )


def _extract_user_id_from_messages_key(key: str) -> Optional[str]:
    """从 messages key 中提取 user_id。

    示例：
        'helix:conversation:user_001:messages' → 'user_001'
    """

    pattern = re.compile(
        r"conversation:(.+):messages$"
    )

    m = pattern.search(key)

    if not m:

        return None

    return m.group(1)


# ============================================================
# TtlRescueScanner
# ============================================================

class TtlRescueScanner:
    """TTL 兜底扫描器。

    线程：单个 daemon 线程，串行处理所有 user。
    异常：每轮扫描捕获全部异常，仅记录日志，不影响下一轮。
    """

    def __init__(
        self,
        scan_interval_sec: int = SCAN_INTERVAL_SEC,
        rescue_window_sec: int = RESCUE_WINDOW_SEC,
        fallback_ttl_sec: int = FALLBACK_TTL_SEC,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> None:

        self._scan_interval_sec = scan_interval_sec
        self._rescue_window_sec = rescue_window_sec
        self._fallback_ttl_sec = fallback_ttl_sec
        self._session_id = session_id

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    # ========================================================
    # 生命周期
    # ========================================================

    def start(self) -> None:
        """启动后台扫描线程。

        重复调用幂等。
        """

        if self._started:

            return

        self._started = True

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="ttl-rescue-scanner",
        )

        self._thread.start()

        logger.info(
            "[TTL-Rescue] 后台扫描线程已启动 | "
            f"interval={self._scan_interval_sec}s | "
            f"window={self._rescue_window_sec}s | "
            f"fallback_ttl={self._fallback_ttl_sec}s"
        )

    def stop(self) -> None:
        """请求停止扫描线程（等当前一轮跑完）。"""

        self._stop_event.set()

        logger.info("[TTL-Rescue] 已请求停止扫描线程")

    # ========================================================
    # 主循环
    # ========================================================

    def _loop(self) -> None:

        # 启动后立即跑一次，之后每 interval 跑一次
        next_run_at = time.time()

        while not self._stop_event.is_set():

            now = time.time()

            if now < next_run_at:

                # 每次 sleep 1 秒，保证 stop 能在 1 秒内响应
                time.sleep(min(1.0, next_run_at - now))
                continue

            try:

                self.scan_one_pass()

            except Exception as e:  # noqa: BLE001

                logger.exception(
                    f"[TTL-Rescue] 本轮扫描失败（不影响下一轮）: {e}"
                )

            next_run_at = time.time() + self._scan_interval_sec

        logger.info("[TTL-Rescue] 扫描线程正常退出")

    # ========================================================
    # Public: 单次完整扫描
    # ========================================================

    def scan_one_pass(self) -> None:
        """执行一次完整扫描。

        单轮扫描：
            - 用 SCAN 枚举所有 messages key
            - 对每个 user 执行 backup（若进入窗口）
            - 对每个 user 执行 rescue（若 messages 已过期）

        抛出异常由 _loop 捕获。
        """

        redis = get_redis()

        if redis is None:

            logger.warning(
                "[TTL-Rescue] Redis 不可用，跳过本轮扫描"
            )

            return

        pattern = _key_messages("*")

        logger.debug(
            f"[TTL-Rescue] 开始扫描 pattern={pattern}"
        )

        users_seen = 0 #扫描到的用户总数
        backed_up = 0  #成功备份的用户数
        rescued = 0    #成功触发兜底抽取的用户数

        cursor = 0     #消息游标

        while True:

            cursor, keys = redis.scan(
                cursor=cursor,
                match=pattern,
                count=100,
            )

            for key in keys:

                key_str = (
                    key.decode()
                    if isinstance(key, bytes)
                    else str(key)
                )

                user_id = _extract_user_id_from_messages_key(
                    key_str
                )

                if not user_id:

                    continue

                users_seen += 1

                # 阶段 A：如果已在兜底窗口，先备份
                did_backup = self._try_backup_user(
                    redis, user_id
                )

                if did_backup:

                    backed_up += 1

                # 阶段 B：如果 messages 已过期，从 fallback 触发抽取
                did_rescue = self._try_rescue_user(
                    redis, user_id
                )

                if did_rescue:

                    rescued += 1

            if cursor == 0:

                break

        logger.info(
            "[TTL-Rescue] 本轮扫描结束 | "
            f"users_seen={users_seen} | "
            f"backed_up={backed_up} | "
            f"rescued={rescued}"
        )

    # ========================================================
    # Internal: 阶段 A —— 备份
    # ========================================================

    def _try_backup_user(
        self,
        redis,
        user_id: str,
    ) -> bool:
        """进入兜底窗口且 fallbackmessages 不存在时，备份待抽取消息。

        返回：是否执行了备份（用于计数）。
        """

        msg_key = _key_messages(user_id)
        fb_key = _key_fallbackmessages(user_id)
        state_key = _key_state(user_id, self._session_id)

        # 1. 检查 messages TTL
        try:

            ttl = redis.ttl(msg_key)

        except Exception as e:  # noqa: BLE001

            logger.warning(
                f"[TTL-Rescue] TTL 查询失败，跳过 | "
                f"user_id={user_id}, error={e}"
            )

            return False

        # 没到窗口：TTL < 0 表示 key 不存在或无过期时间；
        #          TTL >= window 表示还很新鲜
        if ttl < 0 or ttl >= self._rescue_window_sec:

            return False

        # 进入兜底窗口
        logger.debug(
            f"[TTL-Rescue] user={user_id} 进入兜底窗口 | "
            f"ttl={ttl}s"
        )

        # 2. fallbackmessages 已存在则跳过（NX 语义）
        try:

            if redis.exists(fb_key):

                return False

        except Exception as e:  # noqa: BLE001

            logger.warning(
                f"[TTL-Rescue] exists 检查失败，跳过 | "
                f"user_id={user_id}, error={e}"
            )

            return False

        # 3. 取 summary_cursor
        cursor = 0

        try:

            raw = redis.hget(
                state_key, FIELD_SUMMARY_CURSOR
            )

            if raw is not None:

                try:

                    cursor = int(raw)

                except (TypeError, ValueError):

                    cursor = 0

        except Exception as e:  # noqa: BLE001

            logger.warning(
                f"[TTL-Rescue] 读取 summary_cursor 失败，"
                f"按 0 处理 | user_id={user_id}, error={e}"
            )

            cursor = 0

        # 4. 拉取 messages 中 seq > cursor 的项
        pending_dicts: List[dict] = []

        try:

            all_items = redis.lrange(msg_key, 0, -1)

            for raw_item in all_items:

                raw_str = (
                    raw_item.decode()
                    if isinstance(raw_item, bytes)
                    else raw_item
                )

                try:

                    item_dict = json.loads(raw_str)

                except (TypeError, ValueError, json.JSONDecodeError):

                    continue

                seq = _dict_seq(item_dict)

                if seq > cursor:

                    pending_dicts.append(item_dict)

        except Exception as e:  # noqa: BLE001

            logger.exception(
                f"[TTL-Rescue] 读取 messages 失败 | "
                f"user_id={user_id}, error={e}"
            )

            return False

        if not pending_dicts:

            # 没有待抽取内容，仍需标记备份占位，避免反复扫
            payload = "[]"

        else:

            payload = json.dumps(pending_dicts, ensure_ascii=False)

        # 5. 写 fallbackmessages：NX + TTL
        try:

            ok = redis.set(
                fb_key,
                payload,
                nx=True,
                ex=self._fallback_ttl_sec,
            )

        except Exception as e:  # noqa: BLE001

            logger.exception(
                f"[TTL-Rescue] 写 fallbackmessages 失败 | "
                f"user_id={user_id}, error={e}"
            )

            return False

        if not ok:

            logger.debug(
                f"[TTL-Rescue] fallbackmessages NX 竞态已被其他实例写入 | "
                f"user_id={user_id}"
            )

            return False

        logger.info(
            f"[TTL-Rescue] 已备份待抽取消息 | "
            f"user_id={user_id} | "
            f"cursor={cursor} | "
            f"items={len(pending_dicts)} | "
            f"ttl={self._fallback_ttl_sec}s"
        )

        return True

    # ========================================================
    # Internal: 阶段 B —— 兜底抽取
    # ========================================================

    def _try_rescue_user(
        self,
        redis,
        user_id: str,
    ) -> bool:
        """messages 已过期（TTL == -2）时，
        从 fallbackmessages 取出备份并触发记忆抽取。

        返回：是否触发了记忆抽取（用于计数）。
        """

        msg_key = _key_messages(user_id)
        fb_key = _key_fallbackmessages(user_id)

        # 1. messages 仍存在或未过期 → 不触发
        try:

            ttl = redis.ttl(msg_key)

        except Exception as e:  # noqa: BLE001

            logger.warning(
                f"[TTL-Rescue] TTL 查询失败，跳过 rescue | "
                f"user_id={user_id}, error={e}"
            )

            return False

        # TTL != -2（=-1 表示无过期 / > 0 仍存活 / 其他）不触发
        if ttl != -2:

            return False

        # 2. 读 fallbackmessages
        try:

            raw = redis.get(fb_key)

        except Exception as e:  # noqa: BLE001

            logger.exception(
                f"[TTL-Rescue] 读 fallbackmessages 失败 | "
                f"user_id={user_id}, error={e}"
            )

            return False

        if raw is None:

            return False

        if isinstance(raw, bytes):

            raw = raw.decode("utf-8", errors="replace")

        try:

            items = json.loads(raw)

            if not isinstance(items, list):

                items = []

        except (TypeError, ValueError, json.JSONDecodeError) as e:

            logger.warning(
                f"[TTL-Rescue] fallbackmessages 解析失败，丢弃 | "
                f"user_id={user_id}, error={e}"
            )

            # 解析失败 → 删除，避免下次反复读
            try:

                redis.delete(fb_key)

            except Exception:  # noqa: BLE001

                pass

            return False

        if not items:

            # 空备份：直接删，不触发抽取
            try:

                redis.delete(fb_key)

            except Exception:  # noqa: BLE001

                pass

            return False

        # 3. 还原为 BaseMessage 列表
        messages: List[BaseMessage] = []

        for d in items:

            msg = _dict_to_msg(d)

            if msg is not None:

                messages.append(msg)

        if not messages:

            logger.warning(
                f"[TTL-Rescue] fallbackmessages 无法还原任何合法消息 | "
                f"user_id={user_id}"
            )

            try:

                redis.delete(fb_key)

            except Exception:  # noqa: BLE001

                pass

            return False

        # 4. 触发 Memory Graph（后台调用，不阻塞扫描线程本身）
        #    注意：函数内 import，避免扫描模块在容器初始化之前加载时循环依赖
        try:

            from core.memory_graph import memory_graph

        except Exception as e:  # noqa: BLE001

            logger.warning(
                f"[TTL-Rescue] 加载 memory_graph 失败，跳过 rescue | "
                f"user_id={user_id}, error={e}"
            )

            return False

        def _rescue_task():

            try:

                memory_graph.invoke(
                    {
                        "user_id": user_id,
                        "messages": messages,
                    }
                )

                logger.info(
                    f"[TTL-Rescue] 兜底记忆抽取完成 | "
                    f"user_id={user_id} | "
                    f"msgs={len(messages)}"
                )

            except Exception as e:  # noqa: BLE001

                logger.exception(
                    f"[TTL-Rescue] 兜底记忆抽取失败（用户数据未丢失，下次扫描可重试） | "
                    f"user_id={user_id}, error={e}"
                )

            # 抽取结束（无论成功失败）删除 fallbackmessages：
            # - 成功：不需要了
            # - 失败：避免下次扫描反复抛同一批异常（数据其实已过期）
            #   如果后续需要"失败重试"，可以在这里保留 key 并设较短 TTL
            try:

                r = get_redis()

                if r is not None:

                    r.delete(fb_key)

            except Exception:  # noqa: BLE001

                pass

        thread = threading.Thread(
            target=_rescue_task,
            daemon=True,
            name=f"ttl-rescue-run-{user_id}",
        )

        thread.start()

        logger.info(
            f"[TTL-Rescue] 兜底记忆抽取已调度后台执行 | "
            f"user_id={user_id} | "
            f"msgs={len(messages)} | "
            f"thread={thread.name}"
        )

        return True


# ============================================================
# 全局单例
# ============================================================

_scanner_instance: Optional[TtlRescueScanner] = None


def get_ttl_rescue_scanner() -> TtlRescueScanner:
    """获取全局 TTL 兜底扫描器单例。"""

    global _scanner_instance

    if _scanner_instance is None:

        _scanner_instance = TtlRescueScanner()

    return _scanner_instance
