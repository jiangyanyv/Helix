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
    DEFAULT_SCORE_THRESHOLD = _get_float("DEFAULT_SCORE_THRESHOLD",0.8 )

    # ===================== Embedding API =====================
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "dashscope")
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
    EMBEDDING_DIM = _get_int("EMBEDDING_DIM", 1024)
    EMBEDDING_BATCH_SIZE = _get_int("EMBEDDING_BATCH_SIZE", 16)
    EMBEDDING_TIMEOUT = _get_int("EMBEDDING_TIMEOUT", 30)

    # ===================== 对话历史长度控制 =====================
    # MAX_HISTORY_TURNS = _get_int("MAX_HISTORY_TURNS", 4)
    SUMMARY_TRIGGER_THRESHOLD = _get_int("SUMMARY_TRIGGER_THRESHOLD", 20)
    MAX_HISTORY_TURNS = SUMMARY_TRIGGER_THRESHOLD * 2
    SUMMARY_MAX_TOKENS = _get_int("SUMMARY_MAX_TOKENS", 1000)
    HISTORY_CACHE_TTL_SEC = _get_int("HISTORY_CACHE_TTL_SEC", 3600 * 24 * 3)

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

    # ===================== 工具调用（状态B） =====================
    # False = 状态A（当前方案，不调用工具）
    # True  = 状态B（planner 判断 + task_agent 工具调用 + 两阶段流式）
    ENABLE_TOOL_CALLING = os.getenv(
        "ENABLE_TOOL_CALLING", "false"
    ).lower() == "true"

    # 工具调用超时（秒）
    TOOL_CALL_TIMEOUT_SEC = _get_int("TOOL_CALL_TIMEOUT_SEC", 60)

    # ===================== 语音识别（麦克风 / VAD / ASR）=====================
    # 麦克风
    MIC_SAMPLE_RATE = _get_int("MIC_SAMPLE_RATE", 16000)
    MIC_CHANNELS = _get_int("MIC_CHANNELS", 1)
    MIC_BLOCK_SIZE = _get_int("MIC_BLOCK_SIZE", 512)  # 16kHz 下 512 ≈ 32ms
    MIC_DEVICE = os.getenv("MIC_DEVICE")  # None = 系统默认输入设备

    # VAD（Silero）
    # threshold 0.6 偏严格，过滤风扇/键盘等环境噪声（概率多在 0.3~0.5）
    # 阈值：噪声大就调高（0.7），安静环境可降到 0.5
    VAD_THRESHOLD = _get_float("VAD_THRESHOLD", 0.6)
    VAD_MIN_SPEECH_MS = _get_int("VAD_MIN_SPEECH_MS", 250)
    VAD_MAX_SPEECH_MS = _get_int("VAD_MAX_SPEECH_MS", 30000)
    VAD_SILENCE_MS = _get_int("VAD_SILENCE_MS", 800)
    VAD_ECHO_RELEASE_MS = _get_int("VAD_ECHO_RELEASE_MS", 400)
    # 语音开始确认帧数：需连续 N 帧超阈值才触发打断（512@16k 下 3 帧 ≈ 96ms）
    # 调高更抗瞬态噪声，调低打断更灵敏
    # 连续帧确认：抗噪核心参数
    # 3 帧≈96ms（默认，平衡）；5 帧≈160ms（更抗噪，打断略迟）；1 帧=旧行为（最灵敏）
    VAD_SPEECH_START_FRAMES = _get_int("VAD_SPEECH_START_FRAMES", 3)
    # RMS 能量门控：帧能量低于此阈值直接跳过 VAD（抑制风扇/低音量背景）
    # 0.003 ≈ 强风扇声；0.005 ≈ 严格（可能丢轻声说话）；0 即禁用
    VAD_RMS_THRESHOLD = _get_float("VAD_RMS_THRESHOLD", 0.003)
    # 单极高通截止频率（Hz）：压制风扇/空调低频轰鸣
    # 150=温和（保留男声基频 80-180Hz），200=平衡，300=严格（切人声）；0 即禁用
    VAD_HP_CUTOFF_HZ = _get_int("VAD_HP_CUTOFF_HZ", 150)

    # ASR（SenseVoice / FunASR）
    # MODELSCOPE_CACHE 由 .env 注入（funasr 库内部读取），必须为纯 ASCII 路径
    ASR_MODEL = os.getenv("ASR_MODEL", "iic/SenseVoiceSmall")
    ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda") #auto/cpu/cuda
    ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "auto")
    ASR_SAMPLE_RATE = _get_int("ASR_SAMPLE_RATE", 16000)

