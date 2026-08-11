"""ContextBuilder：将 Persona + 3 类记忆格式化为 LLM system prompt 的上下文段。
不负责查询 Memory（查询由 memory_retriever_node 完成后塞进 AgentState）。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from services.context.system_prompt import SYSTEM_PROMPT
from services.memory.retrieved_memory import RetrievedMemory


def _format_dict_readable(data: Dict[str, Any], indent: int = 0) -> str:
    """把 dict 结构化成"缩进 k: v"文本，数组用逗号分隔，子 dict 缩进，LLM比原始JSON更好读。"""
    prefix = "  " * indent
    lines: List[str] = []
    for k, v in data.items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        if isinstance(v, dict):
            inner = _format_dict_readable(v, indent + 1)
            if inner:
                lines.append(f"{prefix}- {k}:")
                lines.append(inner)
        elif isinstance(v, list):
            text_items = [str(x) for x in v if x not in (None, "", [])]
            if text_items:
                lines.append(f"{prefix}- {k}: {', '.join(text_items)}")
        else:
            lines.append(f"{prefix}- {k}: {v}")
    return "\n".join(lines)


class ContextBuilder:
    """构建 LLM 运行上下文。输出 = Persona prompt + 记忆段。"""

    def build(self, retrieved_memory: Optional[RetrievedMemory]) -> str:
        sections: List[str] = []

        # 1. 当前运行时上下文
        sections.append(self._build_runtime_context())

        # 2. Persona
        sections.append(self._build_persona())

        # 3. Memory
        if retrieved_memory:
            memory_block = self._build_memory_context(retrieved_memory)
            if memory_block:
                sections.append(memory_block)
        return "\n\n".join(s for s in sections if s)

    # ============== internal ==============

    def _build_runtime_context(self) -> str:
        """构建当前运行时上下文。

        当前只提供时间信息，后续可以继续扩展：
        - 用户时区
        - 天气
        - 当前环境
        - 摄像头感知结果
        - 设备状态等
        """
        now = datetime.now(ZoneInfo("Asia/Shanghai"))

        weekday_map = {
            0: "星期一",
            1: "星期二",
            2: "星期三",
            3: "星期四",
            4: "星期五",
            5: "星期六",
            6: "星期日",
        }

        return (
            "【当前时间】\n"
            "----------------\n"
            f"{now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{weekday_map[now.weekday()]}\n"
            "时区：UTC+8（Asia/Shanghai）"
        )

    def _build_persona(self) -> str:
        return (
            "【人设/回复风格】\n"
            "----------------\n"
            f"{SYSTEM_PROMPT.strip()}"
        )

    def _build_memory_context(self, memory: RetrievedMemory) -> str:
        blocks: List[str] = []

        # 1) 用户画像（合并原 profile + preference）
        profile = memory.profile or {}
        profile_text = _format_dict_readable(profile, indent=1)
        if profile_text:
            blocks.append(
                "【用户画像 Profile】\n"
                "-------------------\n"
                f"{profile_text}"
            )

        # 2) 重要人物关系
        # 2) 相关人物关系
        rels = memory.relationships or []
        if rels:
            rel_lines: List[str] = []

            for r in rels:
                name = r.get("canonical_name") or r.get("person_id") or "?"
                relation = r.get("relation") or "未知"

                aliases = r.get("aliases_json") or r.get("aliases") or []
                if isinstance(aliases, str):
                    try:
                        aliases = json.loads(aliases)
                    except Exception:
                        aliases = [aliases]

                extra = r.get("extra_json") or r.get("extra") or {}
                if isinstance(extra, str):
                    try:
                        extra = json.loads(extra)
                    except Exception:
                        extra = {}

                # 人物名称
                rel_lines.append(f"  - 人物姓名：{name}")

                # 明确说明关系主体是“用户”
                rel_lines.append(f"    与用户关系：{relation}")

                # 别名
                valid_aliases = [
                    str(a).strip()
                    for a in aliases
                    if a not in (None, "")
                ]
                if valid_aliases:
                    rel_lines.append(
                        f"    别名：{', '.join(valid_aliases)}"
                    )

                # 其他人物信息
                for key, value in extra.items():
                    if value in (None, "", [], {}):
                        continue

                    # list 转成更适合 LLM 阅读的文本
                    if isinstance(value, list):
                        value = "、".join(str(v) for v in value)

                    # dict 简单展开
                    elif isinstance(value, dict):
                        value = ", ".join(
                            f"{k}={v}"
                            for k, v in value.items()
                            if v not in (None, "", [], {})
                        )

                    rel_lines.append(f"    {key}：{value}")

                # 低置信度才显示
                conf = r.get("confidence")
                if conf is not None:
                    try:
                        conf = float(conf)
                        if conf < 0.9:
                            rel_lines.append(f"    置信度：{conf:.2f}")
                    except (TypeError, ValueError):
                        pass

                rel_lines.append("")

            blocks.append(
                "【相关人物 Relationships】\n"
                + "\n".join(rel_lines).rstrip()
            )

        # 3) 相关事件（含情绪/重要度/score，原 EmotionService 已合并进 metadata.emotion）
        episodic = memory.episodic or []
        if episodic:
            epi_lines: List[str] = []
            for e in episodic:
                c = (e.get("content") or "").strip()
                if not c:
                    continue
                meta = []
                ts = e.get("timestamp")
                if ts is not None:
                    meta.append(f"时间: {str(ts)[:19].replace('T',' ')}")
                imp = e.get("importance")
                if imp is not None:
                    try:
                        meta.append(f"重要: {float(imp):.2f}")
                    except Exception:  # noqa: BLE001
                        pass
                sc = e.get("_score")
                if sc is not None:
                    try:
                        meta.append(f"匹配: {float(sc):.2f}")
                    except Exception:  # noqa: BLE001
                        pass
                tags = e.get("tags") or []
                if tags:
                    meta.append("标签: " + ",".join(str(t) for t in tags if t))
                md = e.get("metadata") or {}
                emo = md.get("emotion") if isinstance(md, dict) else None
                if isinstance(emo, dict):
                    t = emo.get("type")
                    it = emo.get("intensity")
                    if t:
                        s = f"情绪: {t}"
                        if it is not None:
                            s += f"({float(it):.2f})"
                        meta.append(s)
                head = "  - " + c
                if meta:
                    head += f"  [{' | '.join(meta)}]"
                epi_lines.append(head)
            if epi_lines:
                blocks.append(
                    "【相关历史事件 Episodic（含情绪）】\n"
                    "--------------------------------\n"
                    + "\n".join(epi_lines)
                )
        if not blocks:
            return ""
        return (
            "【记忆上下文 Memory Context】\n"
            "============================\n"
            + "\n\n".join(blocks)
            + "\n【记忆上下文结束】"
        )
