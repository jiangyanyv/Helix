"""ProfileService：用户画像服务。

职责：
1. 用户画像读取：LRU Cache → MySQL
2. 用户画像写入：MySQL → 更新 LRU Cache
3. 不 preload 全量用户数据
4. 支持 patch 深度合并
5. 自动补齐默认画像结构
6. LRU 命中时避免频繁访问 MySQL
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
from typing import Any, Dict, Optional

from cachetools import LRUCache
from loguru import logger
from sqlalchemy import text

from config import Config
from infrastructure.database.mysql import get_db_session


# ============================================================
# 默认用户画像结构
# ============================================================

DEFAULT_PROFILE_TEMPLATE: Dict[str, Any] = {
    "identity": {
        "name": None,
        "age": None,
        "gender": None,
        "location": None,
        "job": None,
        "avatar": None,
    },
    "background": {},
    "personality": {
        "traits": [],
        "communication_style": None,
    },
    "preferences": {
        "likes": [],
        "dislikes": [],
        "habits": [],
        "chat_prefs": {
            "tone": None,
            "length": None,
            "use_emoji": None,
        },
    },
}


# ============================================================
# 工具函数
# ============================================================

def _deep_merge(
    base: Dict[str, Any],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    """深度合并两个 dict。

    规则：
    - dict + dict：递归合并
    - 其他类型：patch 覆盖 base
    - list：直接替换
    - 不修改原始对象
    """
    result = copy.deepcopy(base)

    for key, value in patch.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def _normalize_profile(
    profile: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """规范化画像结构。

    用默认模板 + 数据库画像进行深度合并，
    确保新版本增加字段后，老用户画像也能自动补齐。
    """
    if not isinstance(profile, dict):
        return copy.deepcopy(DEFAULT_PROFILE_TEMPLATE)

    return _deep_merge(
        copy.deepcopy(DEFAULT_PROFILE_TEMPLATE),
        profile,
    )


# ============================================================
# ProfileService
# ============================================================

class ProfileService:
    """用户画像服务。

    缓存策略：

        LRU Cache
            ↓ miss
        MySQL
            ↓
        回灌 LRU

    写入：

        MySQL
            ↓
        更新 LRU

    不 preload 用户数据。
    """

    def __init__(self) -> None:
        self._cache: LRUCache[str, Dict[str, Any]] = LRUCache(
            maxsize=Config.PROFILE_LRU_MAXSIZE
        )

    # ========================================================
    # Public API
    # ========================================================

    def get(self, user_id: str) -> Dict[str, Any]:
        """获取用户画像。

        缓存命中：
            直接返回 LRU 数据。

        缓存未命中：
            查询 MySQL，并回灌 LRU。

        用户不存在：
            返回默认画像，并写入 LRU。

        注意：
            MySQL 查询异常不会被当成「用户不存在」。
        """
        self._validate_user_id(user_id)

        # ----------------------------------------------------
        # 1. LRU Cache
        # ----------------------------------------------------
        cached = self._cache.get(user_id)

        if cached is not None:
            return copy.deepcopy(cached)

        # ----------------------------------------------------
        # 2. MySQL
        # ----------------------------------------------------
        try:
            profile = self._db_select(user_id)
        except Exception:
            # 数据库异常时，如果未来有旧缓存，可以优先使用。
            # 当前因为 cache miss，所以没有旧缓存。
            logger.exception(
                f"[Profile] 查询 MySQL 失败，user_id={user_id}"
            )
            raise

        # ----------------------------------------------------
        # 3. 用户不存在
        # ----------------------------------------------------
        if profile is None:
            profile = copy.deepcopy(DEFAULT_PROFILE_TEMPLATE)

        else:
            profile = _normalize_profile(profile)

        # ----------------------------------------------------
        # 4. 回灌 LRU
        # ----------------------------------------------------
        self._cache[user_id] = copy.deepcopy(profile)

        return copy.deepcopy(profile)

    def upsert_patch(
        self,
        user_id: str,
        patch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """增量更新用户画像。

        patch 可以是任意子树，例如：

        {
            "identity": {
                "job": "AI Agent Developer"
            }
        }

        或：

        {
            "preferences": {
                "likes": ["LOL", "AI"]
            }
        }

        会自动与当前画像进行 deep merge。
        """
        self._validate_user_id(user_id)

        if not isinstance(patch, dict):
            raise ValueError(
                f"[Profile] patch 必须是 dict，"
                f"收到 {type(patch).__name__}"
            )

        current = self.get(user_id)

        merged = _deep_merge(
            current,
            patch,
        )

        # 确保最终结构完整
        merged = _normalize_profile(merged)

        # 持久化
        self._db_upsert(
            user_id=user_id,
            profile=merged,
        )

        # 更新 LRU
        self._cache[user_id] = copy.deepcopy(merged)

        return copy.deepcopy(merged)

    def replace(
        self,
        user_id: str,
        full_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """全量替换用户画像。

        注意：
            虽然这是 replace，但仍然会自动补齐默认字段，
            防止不同版本 Profile Schema 不一致。
        """
        self._validate_user_id(user_id)

        if not isinstance(full_profile, dict):
            raise ValueError(
                f"[Profile] full_profile 必须是 dict，"
                f"收到 {type(full_profile).__name__}"
            )

        profile = _normalize_profile(full_profile)

        self._db_upsert(
            user_id=user_id,
            profile=profile,
        )

        self._cache[user_id] = copy.deepcopy(profile)

        return copy.deepcopy(profile)

    def clear(self, user_id: str) -> None:
        """删除用户画像，同时清除 LRU。"""
        self._validate_user_id(user_id)

        with get_db_session() as sess:
            sess.execute(
                text(
                    """
                    DELETE FROM profiles
                    WHERE user_id = :uid
                    """
                ),
                {
                    "uid": user_id,
                },
            )

        self._cache.pop(user_id, None)

        logger.info(
            f"[Profile] 清空 user_id={user_id}"
        )

    # ========================================================
    # Cache API
    # ========================================================

    def invalidate(self, user_id: str) -> None:
        """仅清除 LRU，不删除 MySQL 数据。

        适用于：
        - 外部系统修改了 Profile
        - 管理后台修改了 Profile
        - 需要强制下一次从 MySQL 重新读取
        """
        self._validate_user_id(user_id)

        self._cache.pop(user_id, None)

        logger.debug(
            f"[Profile] LRU invalidate user_id={user_id}"
        )

    # ========================================================
    # DB Layer
    # ========================================================

    def _db_select(
        self,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """从 MySQL 查询用户画像。"""
        with get_db_session() as sess:
            row = sess.execute(
                text(
                    """
                    SELECT profile_json
                    FROM profiles
                    WHERE user_id = :uid
                    LIMIT 1
                    """
                ),
                {
                    "uid": user_id,
                },
            ).mappings().first()

        if row is None:
            return None

        profile = row["profile_json"]

        # MySQL JSON 字段可能由驱动直接解析为 dict，
        # 也可能返回字符串。
        if isinstance(profile, str):
            try:
                profile = json.loads(profile)
            except json.JSONDecodeError:
                logger.exception(
                    f"[Profile] profile_json JSON 解析失败，"
                    f"user_id={user_id}"
                )
                raise ValueError(
                    f"[Profile] profile_json 数据损坏，"
                    f"user_id={user_id}"
                )

        if not isinstance(profile, dict):
            raise ValueError(
                f"[Profile] profile_json 必须是 dict，"
                f"user_id={user_id}, "
                f"type={type(profile).__name__}"
            )

        return _normalize_profile(profile)

    def _db_upsert(
        self,
        user_id: str,
        profile: Dict[str, Any],
    ) -> None:
        """MySQL Upsert 用户画像。"""
        if not isinstance(profile, dict):
            raise ValueError(
                "[Profile] profile 必须是 dict"
            )

        # 再次确保 Schema 完整
        profile = _normalize_profile(profile)

        json_str = json.dumps(
            profile,
            ensure_ascii=False,
        )

        now = _dt.datetime.now()

        sql = text(
            """
            INSERT INTO profiles (
                user_id,
                profile_json,
                created_at,
                updated_at
            )
            VALUES (
                :uid,
                :pj,
                :now,
                :now
            )
            ON DUPLICATE KEY UPDATE
                profile_json = VALUES(profile_json),
                updated_at = VALUES(updated_at)
            """
        )

        with get_db_session() as sess:
            sess.execute(
                sql,
                {
                    "uid": user_id,
                    "pj": json_str,
                    "now": now,
                },
            )

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        """校验 user_id。"""
        if not isinstance(user_id, str):
            raise ValueError(
                f"[Profile] user_id 必须是 str，"
                f"收到 {type(user_id).__name__}"
            )

        if not user_id.strip():
            raise ValueError(
                "[Profile] user_id 不能为空"
            )