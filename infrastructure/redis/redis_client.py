"""Redis 客户端。

负责：
1. Redis 客户端单例初始化
2. Redis 连接检测
3. Redis 不可用时返回 None，由上层决定如何降级

注意：
- 不在 import 阶段连接 Redis
- 第一次调用 get_redis() 时才建立连接
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from config import Config


# ===================== Redis 单例 =====================

_redis_client = None
_redis_init_attempted = False


def get_redis():
    """获取 Redis 客户端。

    Redis 连接采用懒加载：
    - 第一次调用时建立连接
    - 连接成功后复用同一个客户端
    - Redis 不可用时返回 None

    Returns:
        Redis 客户端实例，或者 None
    """
    global _redis_client, _redis_init_attempted

    # 已经尝试过初始化，直接返回结果
    if _redis_init_attempted:
        return _redis_client

    _redis_init_attempted = True

    try:
        import redis
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"[Redis] redis 包未安装，Redis 功能不可用: {e}"
        )
        return None

    try:
        kwargs = {
            "host": Config.REDIS_HOST,
            "port": Config.REDIS_PORT,
            "db": Config.REDIS_DB,
            "decode_responses": True,
            "socket_connect_timeout": 3,
            "socket_timeout": 3,
        }

        if Config.REDIS_PASSWORD:
            kwargs["password"] = Config.REDIS_PASSWORD

        client = redis.Redis(**kwargs)

        # 验证连接
        client.ping()

        _redis_client = client

        logger.info("[Redis] Redis 客户端初始化成功")

    except Exception as e:  # noqa: BLE001
        _redis_client = None

        logger.warning(
            f"[Redis] Redis 连接失败，当前不可用: {e}"
        )

    return _redis_client
