from collections import defaultdict
from typing import Any, Callable, Dict, List

from loguru import logger


class EventBus:
    """
    简易事件总线

    后续可替换Redis Stream

    错误隔离：
        publish 时每个订阅者独立 try/except，
        单个订阅者抛异常不会影响后续订阅者执行。
        （例如 INTERRUPT 事件链路中，即使某个订阅者异常，
         audio_queue.clear() / tts.stop() 仍会被执行）
    """

    def __init__(self):

        self.listeners: Dict[str, List[Callable]] = (
            defaultdict(list)
        )

    def subscribe(
        self,
        event_name: str,
        callback: Callable,
    ):

        self.listeners[event_name].append(callback)

    def publish(
        self,
        event_name: str,
        data: Any,
    ):
        """同步发布事件。

        每个订阅者独立隔离异常：
            - 任一订阅者抛异常 → 记录日志，继续执行下一个
            - 全部订阅者执行完毕后返回
            - 不向上抛异常，保证发布方不受订阅者影响
        """

        callbacks = self.listeners.get(event_name)

        if not callbacks:
            return

        for callback in callbacks:

            try:

                callback(data)

            except Exception as e:  # noqa: BLE001

                logger.exception(
                    f"[EventBus] 订阅者执行异常 | "
                    f"event={event_name} | "
                    f"callback={getattr(callback, '__qualname__', callback)} | "
                    f"error={e}"
                )
