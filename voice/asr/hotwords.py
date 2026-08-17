"""ASR 热词修正配置（独立于 config.py）。

用途：对 SenseVoice 输出文本做后处理，把同音/近音误识别词替换回正确词。
典型场景：人设名「爱莉希雅」常被识别为「爱丽西啊 / 爱丽希雅 / 爱莉西雅」等。

配置格式（字符串）：
    标准词1=错词1,错词2|标准词2=错词3,错词4
分隔符：
    组间分隔：| 或换行（\\n）
    组内分隔：,
    标准词与错词之间：=
示例：
    爱莉希雅=爱丽希雅,爱丽西啊,爱莉西雅,爱莉西亚,艾莉希雅,爱丽希亚,艾丽希雅,艾莉西雅

覆盖方式：
    设置环境变量 ASR_HOTWORD 即可覆盖默认 DEFAULT_HOTWORD_RAW。
    支持热加载：ASR_HOTWORD_FILE 指向一个文本文件，每行一条映射。

替换策略：
    候选错词按长度降序匹配（最长优先），避免短词先替换破坏长词。
"""

from __future__ import annotations

import os
import threading
from typing import List, Tuple

# ============================================================
# 默认热词映射（人设名 / 高频专有名词）
# ============================================================

DEFAULT_HOTWORD_RAW = (
    "爱莉希雅=爱丽希雅,爱丽西啊,爱莉西雅,爱莉西亚,艾莉希雅,爱丽希亚,艾丽希雅,艾莉西雅,爱丽西亚"
)


# ============================================================
# 解析
# ============================================================

def parse_hotword_raw(raw: str) -> List[Tuple[str, str]]:
    """解析热词配置字符串为 (错词, 标准词) 列表，按错词长度降序排序。

    支持组间分隔符：| 或换行；
    组内分隔符：,；
    标准词与错词之间：=
    空行 / # 开头的注释行会被忽略。
    """

    if not raw:
        return []

    pairs: List[Tuple[str, str]] = []

    # 组间分隔：先按换行切，再按 | 切，扁平化
    groups: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        groups.extend(line.split("|"))

    for group in groups:
        group = group.strip()
        if not group or group.startswith("#"):
            continue
        if "=" not in group:
            continue
        correct, wrongs = group.split("=", 1)
        correct = correct.strip()
        if not correct:
            continue
        for w in wrongs.split(","):
            w = w.strip()
            if w and w != correct:
                pairs.append((w, correct))

    # 最长错词优先：避免短词先替换、长词匹配不上
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


# ============================================================
# 运行时配置加载（env 优先；支持文件热加载）
# ============================================================

_lock = threading.Lock()
_cached_items: List[Tuple[str, str]] = []
_cached_signature: str = ""


def _compute_signature() -> str:
    """构造当前配置的签名，用于检测是否需要重新解析。"""

    env_val = os.getenv("ASR_HOTWORD", "")
    file_val = ""
    file_path = os.getenv("ASR_HOTWORD_FILE", "")
    if file_path and os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                file_val = fh.read()
        except OSError:
            file_val = ""

    return f"{env_val}||{file_path}||{len(file_val)}||{file_val}"


def load_hotwords(force_reload: bool = False) -> List[Tuple[str, str]]:
    """获取当前生效的 (错词, 标准词) 列表（按长度降序）。

    优先级：
        1) ASR_HOTWORD 环境变量（覆盖默认）
        2) ASR_HOTWORD_FILE 指向的文件（覆盖默认；若 ASR_HOTWORD 同时设置，以文件为准）
        3) DEFAULT_HOTWORD_RAW

    结果按配置签名缓存；签名变化（env 变更 / 文件长度变化）自动重载。
    force_reload=True 强制重算。
    """

    global _cached_items, _cached_signature

    sig = _compute_signature()
    with _lock:
        if not force_reload and sig == _cached_signature and _cached_items:
            return _cached_items

        # 1) env 覆盖
        env_val = os.getenv("ASR_HOTWORD", "")
        if env_val.strip():
            raw = env_val
        else:
            # 2) 文件覆盖
            file_path = os.getenv("ASR_HOTWORD_FILE", "")
            file_content = ""
            if file_path and os.path.isfile(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as fh:
                        file_content = fh.read()
                except OSError:
                    pass
            raw = file_content if file_content.strip() else DEFAULT_HOTWORD_RAW

        _cached_items = parse_hotword_raw(raw)
        _cached_signature = sig
        return _cached_items


def apply_hotwords(text: str, items: List[Tuple[str, str]] = None) -> str:
    """对文本应用热词替换。

    items 为 None 时自动调用 load_hotwords() 取最新配置。
    """

    if not text:
        return text
    if items is None:
        items = load_hotwords()
    if not items:
        return text
    for wrong, correct in items:
        if wrong in text:
            text = text.replace(wrong, correct)
    return text
