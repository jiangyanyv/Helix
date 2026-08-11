from dotenv import load_dotenv
import os


load_dotenv()

'''
全局配置
'''


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v and v.strip() else default


def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v and v.strip() else default


class Config:

    # ===================== LLM =====================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    LLM_MODEL = os.getenv("OPENAI_MODEL_NAME", "DeepSeek-V4-Flash-0731")

    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")
    NVIDIA_MODEL_NAME = os.getenv("NVIDIA_MODEL_NAME")

    TAVILY_KEY = os.getenv("TAVILY_KEY")

    # ===================== Redis（对话历史/上下文缓存） =====================
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = _get_int("REDIS_PORT", 6379)
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
    REDIS_DB = _get_int("REDIS_DB", 0)
    REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "helix")

    # ===================== MySQL（长期记忆持久化） =====================
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = _get_int("MYSQL_PORT", 3306)
    MYSQL_USER = os.getenv("MYSQL_USER", "helix")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "helix123")
    MYSQL_DB = os.getenv("MYSQL_DB", "helix_db")
    MYSQL_CHARSET = "utf8mb4"

    @classmethod
    def mysql_url(cls) -> str:
        """SQLAlchemy URL"""
        return (
            f"mysql+pymysql://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}"
            f"@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DB}"
            f"?charset={cls.MYSQL_CHARSET}"
        )

    # ===================== Qdrant（向量检索） =====================
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = _get_int("QDRANT_PORT", 6333)
    QDRANT_GRPC_PORT = _get_int("QDRANT_GRPC_PORT", 6334)
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
    QDRANT_USE_GRPC = os.getenv("QDRANT_USE_GRPC", "false").lower() == "true"

    # 集合名称（目前仅 episodic 用向量检索）
    QDRANT_COLLECTION_EPISODIC = "episodic"

    # ===================== Embedding API =====================
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "dashscope")
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
    EMBEDDING_DIM = _get_int("EMBEDDING_DIM", 1024)
    EMBEDDING_BATCH_SIZE = _get_int("EMBEDDING_BATCH_SIZE", 16)
    EMBEDDING_TIMEOUT = _get_int("EMBEDDING_TIMEOUT", 30)

    # ===================== 对话历史长度控制 =====================
    MAX_HISTORY_TURNS = _get_int("MAX_HISTORY_TURNS", 20)
    SUMMARY_TRIGGER_THRESHOLD = _get_int("SUMMARY_TRIGGER_THRESHOLD", 7)
    SUMMARY_MAX_TOKENS = _get_int("SUMMARY_MAX_TOKENS", 500)
    HISTORY_CACHE_TTL_SEC = _get_int("HISTORY_CACHE_TTL_SEC", 3600 * 24 * 7)

    # ===================== 缓存策略（Memory Service用） =====================
    PROFILE_LRU_MAXSIZE = _get_int("PROFILE_LRU_MAXSIZE", 50)
    RELATIONSHIP_LRU_MAXSIZE = _get_int("RELATIONSHIP_LRU_MAXSIZE", 50)
    RELATIONSHIP_ALIAS_TTL_SEC = _get_int("RELATIONSHIP_ALIAS_TTL_SEC", 3600)
    EPISODIC_QUERY_CACHE_MAXSIZE = _get_int("EPISODIC_QUERY_CACHE_MAXSIZE", 100)
    EPISODIC_QUERY_CACHE_TTL_SEC = _get_int("EPISODIC_QUERY_CACHE_TTL_SEC", 300)
    SEMANTIC_SNAPSHOT_REFRESH_SEC = _get_int("SEMANTIC_SNAPSHOT_REFRESH_SEC", 6 * 3600)
    MAX_EPISODIC_TOP_K = _get_int("MAX_EPISODIC_TOP_K", 10)

    # ===================== 管理接口 =====================
    ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN")

