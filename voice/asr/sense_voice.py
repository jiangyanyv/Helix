"""SenseVoice ASR 识别模块（基于 FunASR）。

将 VAD 切分出的语音段送入 SenseVoice 推理，输出结构化 VoiceInput。
支持解析 SenseVoice 的标签格式（语种 / 情绪 / 事件）。

重型依赖（funasr / torch）懒加载；MODELSCOPE_CACHE 必须为纯 ASCII 路径。
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
from loguru import logger

from voice.models import EmotionTag, Language, VoiceInput

# SenseVoice 输出标签正则：形如 <|zh|>
_TAG_RE = re.compile(r"<\|([^|]+)\|>")

# 语种标签映射（小写归一）
_LANGUAGE_MAP = {
    "zh": Language.ZH,
    "en": Language.EN,
    "ja": Language.JA,
    "ko": Language.KO,
    "yue": Language.YUE,
}

# 情绪标签映射（大写）
_EMOTION_MAP = {
    "HAPPY": EmotionTag.HAPPY,
    "SAD": EmotionTag.SAD,
    "ANGRY": EmotionTag.ANGRY,
    "NEUTRAL": EmotionTag.NEUTRAL,
    "FEARFUL": EmotionTag.FEARFUL,
    "DISGUSTED": EmotionTag.DISGUSTED,
    "SURPRISED": EmotionTag.SURPRISED,
}

# 事件标签集合
_EVENT_TAGS = {"Speech", "BGM", "Laughter", "Applause"}

# 格式控制标签（忽略），如 woitn / withitn
_IGNORE_TAGS = {"woitn", "withitn"}


class SenseVoiceASR:
    """基于 FunASR + SenseVoice 的语音识别器。"""

    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cuda",
        language: str = "auto",
        sample_rate: int = 16000,
    ):
        self.model_name = model_name
        self.language = language
        self.sample_rate = sample_rate

        # 解析实际可用 device（cuda 不可用时回退 cpu）
        self.device = self._resolve_device(device)

        self._model = None

    # ==================================================
    # 设备解析（带 CUDA 可用性回退）
    # ==================================================

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning(
                        "[SenseVoiceASR] ASR_DEVICE=cuda 但 CUDA 不可用，"
                        "回退到 cpu"
                    )
                    return "cpu"
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"[SenseVoiceASR] 检测 CUDA 可用性失败，回退 cpu: {e}"
                )
                return "cpu"
        return device

    # ==================================================
    # 本地模型路径解析（跳过 ModelScope 联网检查）
    # ==================================================

    def _resolve_local_model_path(
        self,
        model_name: str,
    ) -> Optional[str]:
        """解析 ModelScope 本地缓存路径。

        缓存结构：
            {MODELSCOPE_CACHE}/models/{repo_dir (/ -> --)}/snapshots/{rev}/
        若该目录下存在 model.pt，则返回该快照目录路径；否则返回 None。
        """
        cache_root = os.getenv("MODELSCOPE_CACHE")
        if not cache_root:
            return None

        repo_dir = model_name.replace("/", "--")
        models_dir = os.path.join(cache_root, "models", repo_dir)
        snapshots_dir = os.path.join(models_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return None

        # 优先 master，其次任意快照目录
        candidates = []
        master_dir = os.path.join(snapshots_dir, "master")
        if os.path.isdir(master_dir):
            candidates.append(master_dir)
        for name in os.listdir(snapshots_dir):
            full = os.path.join(snapshots_dir, name)
            if os.path.isdir(full) and full not in candidates:
                candidates.append(full)

        for cand in candidates:
            if os.path.isfile(os.path.join(cand, "model.pt")):
                return cand
        return None

    # ==================================================
    # 模型懒加载
    # ==================================================

    def load(self):
        """懒加载 FunASR 模型。重入安全。"""
        if self._model is not None:
            return

        # 防御性确保 MODELSCOPE_CACHE 已设置（funasr 库内部也会读取）
        cache_root = os.getenv("MODELSCOPE_CACHE")
        if cache_root:
            os.environ.setdefault("MODELSCOPE_CACHE", cache_root)

        local_path = self._resolve_local_model_path(self.model_name)

        if local_path:
            load_target = local_path
            source = "local"
        else:
            # 无本地模型，需从 ModelScope hub 下载。
            # 此时 MODELSCOPE_CACHE 必须为纯 ASCII 路径，否则 funasr 会用
            # 默认缓存（Windows 下含中文用户名），导致 bpe.model 等加载失败。
            if not cache_root:
                raise RuntimeError(
                    "[SenseVoiceASR] 未找到本地模型且未设置 MODELSCOPE_CACHE。"
                    "请设置环境变量 MODELSCOPE_CACHE 为纯 ASCII 路径，"
                    "或将模型放置于 {MODELSCOPE_CACHE}/models/ 下。"
                )
            load_target = self.model_name
            source = "hub"

        logger.info(
            f"[SenseVoiceASR] 加载模型 load_target={load_target} "
            f"device={self.device} source={source}"
        )

        from funasr import AutoModel

        self._model = AutoModel(
            model=load_target,
            device=self.device,
            disable_pbar=True,
            disable_update=True,
            trust_remote_code=True,
        )
        logger.info("[SenseVoiceASR] 模型加载完成")

    # ==================================================
    # 输出解析（纯函数，可独立测试）
    # ==================================================

    @staticmethod
    def parse_sensevoice_output(raw: str) -> dict:
        """解析 SenseVoice 标签输出。

        格式示例：<|zh|><|HAPPY|><|Speech|><|woitn|>今天天气真好
        返回：
            {
              "text": 纯文本（已 strip）,
              "language": Language,
              "emotion": EmotionTag,
              "event": str | None,
            }
        """
        language = Language.UNKNOWN
        emotion = EmotionTag.UNKNOWN
        event: Optional[str] = None

        for tag in _TAG_RE.findall(raw):
            key = tag.strip()

            low = key.lower()
            if low in _LANGUAGE_MAP:
                language = _LANGUAGE_MAP[low]
                continue

            up = key.upper()
            if up in _EMOTION_MAP:
                emotion = _EMOTION_MAP[up]
                continue

            if key in _EVENT_TAGS:
                event = key
                continue

            if low in _IGNORE_TAGS:
                continue

        text = _TAG_RE.sub("", raw).strip()
        return {
            "text": text,
            "language": language,
            "emotion": emotion,
            "event": event,
        }

    # ==================================================
    # 推理
    # ==================================================

    def transcribe(self, audio: np.ndarray) -> VoiceInput:
        """对一段语音（float32 numpy）进行识别，返回 VoiceInput。"""
        self.load()

        audio = np.ascontiguousarray(audio).astype(np.float32)

        results = self._model.generate(
            input=audio,
            language=self.language,
            use_itn=True,
            sample_rate=self.sample_rate,
        )

        raw_text = ""
        if results:
            first = results[0]
            if isinstance(first, dict):
                raw_text = first.get("text", "")
            else:
                # 部分 funasr 版本返回对象
                raw_text = getattr(first, "text", "") or str(first)

        parsed = self.parse_sensevoice_output(raw_text)

        # BGM 事件不计为有效文本
        text = "" if parsed["event"] == "BGM" else parsed["text"]
        text_stripped = text.strip()

        confidence = 0.95 if text_stripped else 0.0
        duration = len(audio) / float(self.sample_rate)

        return VoiceInput(
            text=text_stripped,
            emotion=parsed["emotion"],
            language=parsed["language"],
            confidence=confidence,
            duration=duration,
            is_final=True,
        )
