"""麦克风采集模块。

sounddevice 输入流回调（独立音频线程）→ call_soon_threadsafe 桥接到
asyncio 队列，由协程消费。所有重型依赖懒加载。
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from loguru import logger


class Microphone:
    """基于 sounddevice 的麦克风采集器。

    输出每帧为 float32 单声道 numpy 数组，已归一化到 [-1, 1]。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 512,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device = device

        self._stream = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None

    # ==================================================
    # 帧入队（在事件循环线程中执行）
    # ==================================================

    def _safe_put(self, frame):
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning(
                "[Microphone] 音频帧队列已满，丢弃当前帧"
                f"（队列上限 {self._queue.maxsize}）"
            )

    # ==================================================
    # sounddevice 回调（在独立音频线程中执行）
    # ==================================================

    def _callback(self, indata, frames, time_info, status):
        if status.input_overflow:
            logger.warning("[Microphone] 输入溢出（input_overflow），部分采样丢失")

        # indata 形状: (frames, channels)，dtype=int16
        mono_int16 = indata[:, 0]
        mono_float32 = mono_int16.astype("float32") / 32768.0

        # 桥接到事件循环所在线程
        if self._loop is not None and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._safe_put, mono_float32)
            except RuntimeError:
                # 事件循环已关闭
                pass

    # ==================================================
    # 流式输出
    # ==================================================

    async def stream(self) -> AsyncIterator:
        """异步产出音频帧（float32 单声道 numpy 数组）。

        在 finally 中关闭输入流，外部也可调用 stop() 同步关闭。
        """
        import numpy as np  # noqa: F401  确保 numpy 可用
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=200)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.block_size,
            dtype="int16",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            f"[Microphone] 采集已启动 "
            f"sample_rate={self.sample_rate} "
            f"block_size={self.block_size} "
            f"device={self.device}"
        )

        try:
            while True:
                frame = await self._queue.get()
                if frame is None:
                    break
                yield frame
        finally:
            self._close_stream()

    # ==================================================
    # 停止 / 关闭
    # ==================================================

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[Microphone] 停止流时忽略异常: {e}")
            try:
                self._stream.close()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[Microphone] 关闭流时忽略异常: {e}")
            self._stream = None

    def stop(self):
        """同步停止采集并唤醒等待中的消费者。"""
        self._close_stream()
        if self._queue is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            except RuntimeError:
                pass
