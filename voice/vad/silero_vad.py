"""Silero VAD 流式语音分段。

消费麦克风帧流，按语音/静音边界切分出完整语音段。
内置回声抑制：TTS 播放期间及停止后一段释放期内丢弃帧，
避免把 TTS 输出误识别为用户语音。

重型依赖（torch / silero_vad）懒加载，import 放在方法内部。
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, List, Optional

import numpy as np
from loguru import logger


class SileroVAD:
    """基于 Silero 的流式 VAD 分段器。

    帧处理流水线（按顺序）：
        1) 单极高通滤波器（压制风扇/空调 <200Hz 低频轰鸣）
        2) RMS 能量门控（低能量连续噪声直接判静音，不送入 Silero 模型）
        3) Silero 模型 → 单帧语音概率 → 分段状态机 → 连续帧确认
    """

    def __init__(
        self,
        threshold: float = 0.6,
        min_speech_ms: int = 250,
        max_speech_ms: int = 30000,
        silence_ms: int = 600,
        sample_rate: int = 16000,
        echo_release_ms: int = 400,
        speech_start_frames: int = 3,
        rms_threshold: float = 0.003,
        hp_cutoff_hz: int = 150,
    ):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.silence_ms = silence_ms
        self.sample_rate = sample_rate
        self.echo_release_ms = echo_release_ms
        # 连续语音帧确认：需连续 N 帧超阈值才触发 on_speech_start（打断 TTS），
        # 避免瞬态噪声（风扇/键盘）单帧误触发。512@16k 下 3 帧 ≈ 96ms
        self.speech_start_frames = speech_start_frames
        # RMS 能量门控：低于此阈值视为「背景噪声」，直接跳过 VAD 模型。
        # 经验值：风扇声 RMS ≈ 0.002~0.008，轻声说话 >= 0.01
        self.rms_threshold = rms_threshold
        # 单极高通截止频率（Hz）：>150 压风扇，>250 会轻微削弱人声基频
        self.hp_cutoff_hz = hp_cutoff_hz

        # 高通滤波器状态（y[i] = x[i] - x[i-1] + alpha * y[i-1]）
        self._hp_alpha: Optional[float] = None
        self._hp_x_prev: float = 0.0
        self._hp_y_prev: float = 0.0

        self._model = None
        self._torch = None

        # 回声抑制：由外部注入「TTS 是否正在播放」的查询
        self._is_output_playing: Optional[Callable[[], bool]] = None
        # 语音开始回调（用于打断 TTS）
        self.on_speech_start: Optional[Callable[[], None]] = None

    # ==================================================
    # 模型懒加载
    # ==================================================

    def load(self):
        """懒加载 silero_vad 模型。重入安全。"""
        if self._model is not None:
            return

        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad()
        logger.info("[SileroVAD] 模型加载完成")

    # ==================================================
    # 回声抑制注入
    # ==================================================

    def set_echo_check(self, check: Callable[[], bool]):
        """注入「TTS 是否正在播放」的查询函数，用于回声抑制。"""
        self._is_output_playing = check

    def set_on_speech_start(self, callback: Callable[[], None]):
        """注入语音开始回调（通常用于打断 TTS）。"""
        self.on_speech_start = callback

    # ==================================================
    # 单帧语音概率
    # ==================================================

    def _speech_prob(self, frame: np.ndarray) -> float:
        torch = self._torch
        tensor = torch.from_numpy(
            np.ascontiguousarray(frame, dtype=np.float32)
        )
        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate).item()
        return float(prob)

    # ==================================================
    # 缓冲区输出
    # ==================================================

    @staticmethod
    def _emit(buffer: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(buffer).astype(np.float32)

    # ==================================================
    # 流式分段状态机
    # ==================================================

    async def segments(
        self,
        frames: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[np.ndarray]:
        """消费帧流，按语音/静音边界产出完整语音段（float32 numpy 数组）。"""
        self.load()

        ms_per_sample = 1000.0 / self.sample_rate
        echo_release_samples = int(
            self.sample_rate * self.echo_release_ms / 1000.0
        )

        speech_buffer: List[np.ndarray] = []
        in_speech = False
        speech_samples = 0
        silence_samples = 0

        # on_speech_start 连续帧确认状态
        speech_start_confirmed = False
        speech_start_consecutive = 0

        # 回声抑制状态
        was_playing = False
        post_play_samples = 0

        def reset_segment():
            nonlocal speech_buffer, in_speech, speech_samples, silence_samples
            nonlocal speech_start_confirmed, speech_start_consecutive
            speech_buffer = []
            in_speech = False
            speech_samples = 0
            silence_samples = 0
            speech_start_confirmed = False
            speech_start_consecutive = 0

        async for frame in frames:
            frame_len = len(frame)

            # ----------------------------------------------
            # 1) 回声抑制：TTS 正在播放 → 丢弃帧 + 清空 + 重置
            # ----------------------------------------------
            if (
                self._is_output_playing is not None
                and self._is_output_playing()
            ):
                if speech_buffer:
                    reset_segment()
                try:
                    self._model.reset_states()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[SileroVAD] 回声重置忽略异常: {e}")
                was_playing = True
                post_play_samples = 0
                continue

            # ----------------------------------------------
            # 2) TTS 刚停止的释放期 → 仍丢弃帧，避免尾音回声
            # ----------------------------------------------
            if was_playing:
                post_play_samples += frame_len
                if post_play_samples < echo_release_samples:
                    continue
                # 释放期结束，正式恢复 VAD
                was_playing = False
                try:
                    self._model.reset_states()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[SileroVAD] 释放期重置忽略异常: {e}")

            # ----------------------------------------------
            # 3) 先抑噪再判别
            #    3a. 单极高通滤波（压制风扇 <200Hz 低频轰鸣）
            #    3b. RMS 能量门控（低能量背景噪声直接判静音，跳过 Silero 模型）
            # ----------------------------------------------

            # 3a. 高通滤波
            if self._hp_alpha is None:
                # 单极 HPF: alpha = exp(-2*pi*fc/fs)
                self._hp_alpha = float(
                    np.exp(-2.0 * np.pi * self.hp_cutoff_hz / self.sample_rate)
                )

            alpha = self._hp_alpha
            x_prev = self._hp_x_prev
            y_prev = self._hp_y_prev
            filtered = np.empty_like(frame)
            for n in range(frame_len):
                x_n = float(frame[n])
                y_n = x_n - x_prev + alpha * y_prev
                filtered[n] = y_n
                x_prev = x_n
                y_prev = y_n
            self._hp_x_prev = x_prev
            self._hp_y_prev = y_prev
            frame = filtered

            # 3b. RMS 能量门控
            if self.rms_threshold > 0.0:
                # np.sqrt(np.mean(frame*frame)) 用 float32 快速实现
                rms = float(
                    np.sqrt(np.mean(np.square(frame, dtype=np.float32)))
                )
                if rms < self.rms_threshold:
                    # 视为静音，跳过模型和分段累积
                    if not speech_start_confirmed:
                        speech_start_consecutive = 0
                    # 静音行为：若在语音段则累计静音采样，否则无副作用
                    if in_speech:
                        silence_samples += frame_len
                        speech_buffer.append(frame)
                        if silence_samples * ms_per_sample >= self.silence_ms:
                            if (
                                speech_samples * ms_per_sample
                                >= self.min_speech_ms
                            ):
                                yield self._emit(speech_buffer)
                            reset_segment()
                            try:
                                self._model.reset_states()
                            except Exception as e:  # noqa: BLE001
                                logger.debug(
                                    f"[SileroVAD] RMS门控段尾重置忽略异常: {e}"
                                )
                    continue

            # ----------------------------------------------
            # 4) 常规 VAD 判定
            # ----------------------------------------------
            try:
                prob = self._speech_prob(frame)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[SileroVAD] 单帧判定失败，跳过该帧: {e}")
                continue

            if prob >= self.threshold:
                # 语音帧
                if not in_speech:
                    in_speech = True
                    silence_samples = 0

                speech_buffer.append(frame)
                speech_samples += frame_len

                # on_speech_start 连续帧门控：需连续 N 帧超阈值才触发打断，
                # 避免瞬态噪声（风扇/键盘）单帧误触发。
                # 注意：分段逻辑（in_speech / speech_buffer）不受此门控影响，
                # 仅延迟打断回调的触发时机。
                if not speech_start_confirmed:
                    speech_start_consecutive += 1
                    if speech_start_consecutive >= self.speech_start_frames:
                        speech_start_confirmed = True
                        if self.on_speech_start is not None:
                            try:
                                self.on_speech_start()
                            except Exception as e:  # noqa: BLE001
                                logger.warning(
                                    f"[SileroVAD] on_speech_start 回调异常: {e}"
                                )

                # 单句最长 → 强制截断
                if speech_samples * ms_per_sample >= self.max_speech_ms:
                    yield self._emit(speech_buffer)
                    reset_segment()
                    try:
                        self._model.reset_states()
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"[SileroVAD] 截断后重置忽略异常: {e}")
            else:
                # 静音帧
                # 语音开始未确认时，连续帧被打破，重置计数
                if not speech_start_confirmed:
                    speech_start_consecutive = 0

                if in_speech:
                    silence_samples += frame_len
                    speech_buffer.append(frame)

                    # 句尾静音达阈值 → 结束本段
                    if silence_samples * ms_per_sample >= self.silence_ms:
                        if (
                            speech_samples * ms_per_sample
                            >= self.min_speech_ms
                        ):
                            yield self._emit(speech_buffer)
                        # 否则视为过短噪声，丢弃
                        reset_segment()
                        try:
                            self._model.reset_states()
                        except Exception as e:  # noqa: BLE001
                            logger.debug(
                                f"[SileroVAD] 句尾后重置忽略异常: {e}"
                            )
