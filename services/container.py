import os

from loguru import logger

from services.llm.deepseek_client import DeepSeekClient
from services.response.response_service import ResponseService
from core.context.context_builder import ContextBuilder
from core.context.message_builder import MessageBuilder
from services.event.event_bus import EventBus
from voice.queue.audio_queue import AudioQueue
from core.runtime.runtime_manager import RuntimeManager
from voice.tts.tts_worker import TTSWorker
from core.runtime.event_handler import RuntimeEventHandler
from core.session.conversation_manager import ConversationManager
from core.session.summarizer import Summarizer

# ===================== Memory: 3 个新 Service =====================
from services.memory.profile_service import ProfileService
from services.memory.relationship_service import RelationshipService
from services.memory.episodic_service import EpisodicService

# ===================== Memory Pipeline =====================
from memory.memory_pipeline.memory_extractor import MemoryExtractor
from memory.memory_pipeline.memory_judge import MemoryJudge
from memory.memory_pipeline.memory_updater import MemoryUpdater

# ===================== Embedding + Vector Store（可选，失败降级） =====================


class ServiceContainer:
    """
    全局服务容器

    管理所有单例Service，确保全链路共享同一组实例
    （避免 App、Graph、各Node 各自 new 不同的 Queue / Runtime）

    初始化策略：
    - 必选组件（LLM/Runtime/Audio/3个基础Memory Service）：失败直接抛异常
    - 可选组件（Embedding / Qdrant）：配置缺失或初始化失败 → 打 warning + 降级
      （Episodic 走 search_recent 最近 N 条路径，不影响主流程）
    """

    def __init__(self):

        # =====================
        # Event Bus（最先初始化，后面都要订阅）
        # =====================
        self.event_bus = EventBus()

        # =====================
        # Runtime
        # =====================
        self.runtime_manager = RuntimeManager()
        self.turn_manager = self.runtime_manager.turn_manager

        # =====================
        # Audio Queue + TTS
        # =====================
        self.audio_queue = AudioQueue()
        self.tts_worker = TTSWorker(
            self.audio_queue,
            self.runtime_manager
        )

        # =====================
        # Memory - 3 个核心 Service（必选，MySQL 未启动时写操作才会失败）
        # =====================
        self.profile_service = ProfileService()
        self.relationship_service = RelationshipService()
        self.episodic_service = EpisodicService()

        # # 兼容别名（避免外部旧引用报错）
        # self.profile = self.profile_service
        # self.episodic = self.episodic_service
        # self.relationship = self.relationship_service

        # =====================
        # Embedding + Vector Store（可选，失败降级）
        # =====================
        self.embedding_provider = None
        self.vector_store = None
        try:
            from services.embedding.dashscope_provider import DashScopeEmbeddingProvider
            self.embedding_provider = DashScopeEmbeddingProvider()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"[Container] Embedding 初始化失败（跳过向量链路，Episodic 将降级最近N条）: {e}"
            )
            self.embedding_provider = None

        if self.embedding_provider is not None:
            try:
                from infrastructure.vector.qdrant_store import QdrantVectorStore
                self.vector_store = QdrantVectorStore()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"[Container] Qdrant 初始化失败（跳过向量链路，Episodic 将降级最近N条）: {e}"
                )
                self.vector_store = None

        # 把 Embedding + VectorStore 注入进 EpisodicService
        if self.embedding_provider is not None and self.vector_store is not None:
            try:
                self.episodic_service.bind_external(
                    embedding_provider=self.embedding_provider,
                    vector_store=self.vector_store,
                )
                # 不在这里立即 ensure_collection（Qdrant 可能尚未启动），留给首次 search 时触发
                # logger.info("[Container] EpisodicService 已绑定 Embedding + Qdrant")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[Container] Episodic bind_external 失败: {e}")

        # =====================
        # Memory Pipeline（Extractor / Judge / Updater）
        # Updater 依赖 3 个 Memory Service，通过构造函数显式注入
        # =====================
        self.memory_extractor = MemoryExtractor()
        self.memory_judge = MemoryJudge()
        self.memory_updater = MemoryUpdater(
            profile_service=self.profile_service,
            relationship_service=self.relationship_service,
            episodic_service=self.episodic_service,
        )

        # =====================
        # Session / 对话历史（Redis + 滑动窗口 + 滚动摘要）
        # Summarizer 注入 ConversationManager，触发摘要时调用 LLM
        # =====================
        self.summarizer = Summarizer()

        self.conversation_manager = ConversationManager(
            summarizer=self.summarizer,
        )

        # =====================
        # LLM
        # =====================
        self.llm_client = DeepSeekClient()

        # =====================
        # Response
        # =====================
        self.response_service = ResponseService(
            self.llm_client
        )

        # =====================
        # Context
        # =====================
        self.context_builder = ContextBuilder()
        self.message_builder = MessageBuilder()

        # =====================
        # Runtime Event Handler
        # 把事件总线 -> Runtime / TTS 的状态切换连接起来
        # =====================
        self.event_handler = RuntimeEventHandler(
            runtime_manager=self.runtime_manager,
            audio_queue=self.audio_queue,
            tts_worker=self.tts_worker,
            event_bus=self.event_bus
        )

        # =====================
        # 启动健康检查（必选依赖：Redis / MySQL）
        #
        # Embedding / Qdrant 已在上方降级处理，不在此检查范围。
        # 失败直接抛异常，让部署方在启动时立即发现问题，
        # 而不是延迟到首次调用才暴露。
        #
        # 开发/测试环境可通过环境变量跳过：
        #   HELIX_SKIP_HEALTH_CHECK=true
        # =====================
        self._health_check()

    # ==================================================
    # 启动健康检查
    # ==================================================

    def _health_check(self) -> None:
        """启动时检查必选依赖（Redis / MySQL）。

        - Redis：ConversationManager 唯一存储，不可降级
        - MySQL：3 个 Memory Service 的事实源，不可降级
        - Embedding / Qdrant：可选，已在 __init__ 中降级处理

        失败时抛 RuntimeError，避免运行时才暴露问题。
        """

        if os.getenv(
            "HELIX_SKIP_HEALTH_CHECK", ""
        ).lower() == "true":

            logger.warning(
                "[Container] 跳过启动健康检查"
                "（HELIX_SKIP_HEALTH_CHECK=true）"
            )

            return

        # =====================
        # Redis
        # =====================

        from infrastructure.redis.redis_client import get_redis

        redis_client = get_redis()

        if redis_client is None:

            raise RuntimeError(
                "[Container] Redis 健康检查失败：连接不可用。"
                "请检查 Redis 服务是否启动、配置是否正确。"
                "如需跳过检查，"
                "设置环境变量 HELIX_SKIP_HEALTH_CHECK=true"
            )

        logger.info("[Container] Redis 健康检查通过")

        # =====================
        # MySQL
        # =====================

        try:

            from infrastructure.database.mysql import (
                create_mysql_engine,
            )
            from sqlalchemy import text

            engine = create_mysql_engine()

            with engine.connect() as conn:

                conn.execute(text("SELECT 1"))

            logger.info("[Container] MySQL 健康检查通过")

        except Exception as e:

            raise RuntimeError(
                f"[Container] MySQL 健康检查失败：{e}。"
                "请检查 MySQL 服务是否启动、配置是否正确。"
                "如需跳过检查，"
                "设置环境变量 HELIX_SKIP_HEALTH_CHECK=true"
            ) from e


# 全局单例
container = ServiceContainer()
