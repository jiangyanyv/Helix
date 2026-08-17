"""语音识别管线：麦克风采集 → Silero VAD 分段 → SenseVoice ASR 识别。

组合 Microphone / SileroVAD / SenseVoiceASR 三个单例，对外暴露
统一的生命周期与异步流式接口。ASR 推理放到线程池避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Optional

from loguru import logger

from voice.asr.sense_voice import SenseVoiceASR
from voice.capture.microphone import Microphone
from voice.models import VoiceInput
from voice.vad.silero_vad import SileroVAD


class VoicePipeline:
    """麦克风 → VAD → ASR 的异步管线。"""

    def __init__(
        self,
        microphone: Microphone,
        vad: SileroVAD,
        asr: SenseVoiceASR,
    ):
        self.microphone = microphone
        self.vad = vad
        self.asr = asr

    # ==================================================
    # 生命周期
    # ==================================================

    async def _on_start(self):
        """预加载模型（已加载则跳过），放到线程池避免阻塞事件循环。"""
        if self.vad._model is None:
            await asyncio.to_thread(self.vad.load)
        if self.asr._model is None:
            await asyncio.to_thread(self.asr.load)

    def _on_stop(self):
        """停止麦克风采集。"""
        self.microphone.stop()

    # ==================================================
    # 回调透传
    # ==================================================

    def set_on_speech_start(self, callback: Callable[[], None]):
        """注入语音开始回调（用于打断 TTS）。"""
        self.vad.set_on_speech_start(callback)

    def set_echo_check(self, check: Callable[[], bool]):
        """注入「TTS 是否正在播放」查询，用于回声抑制。"""
        self.vad.set_echo_check(check)

    # ==================================================
    # 主流程
    # ==================================================

    async def stream(self) -> AsyncIterator[VoiceInput]:
        """启动管线并流式产出识别结果 VoiceInput。

        会先预加载模型，随后进入采集 → 分段 → 识别循环。
        finally 中关闭麦克风帧流。
        """
        await self._on_start()

        frame_stream = self.microphone.stream()
        try:
            async for segment_audio in self.vad.segments(frame_stream):
                voice_input: VoiceInput = await asyncio.to_thread(
                    self.asr.transcribe, segment_audio
                )
                if not voice_input.text.strip():
                    continue
                yield voice_input
        finally:
            # 关闭麦克风帧流（aiter 不保证 close，显式停止采集）
            self._on_stop()
