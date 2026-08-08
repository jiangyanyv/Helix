from services.llm.deepseek_client import DeepSeekClient
from services.response.response_service import ResponseService
from services.context.context_builder import ContextBuilder
from services.context.message_builder import MessageBuilder
from services.memory.memory_manager import MemoryManager
from services.event.event_bus import EventBus
from voice.queue.audio_queue import AudioQueue
from core.runtime.runtime_manager import RuntimeManager
from voice.tts.tts_worker import TTSWorker
from core.runtime.event_handler import RuntimeEventHandler


class ServiceContainer:
    """
    全局服务容器

    管理所有单例Service，确保全链路共享同一组实例
    （避免 App、Graph、各Node 各自 new 不同的 Queue / Runtime）
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

        # 方便直接访问 turn_manager
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
        # Memory
        # =====================
        self.memory_manager = (
            MemoryManager()
        )

        # 方便直接引用各 Memory Service（MemoryUpdater 内部会用到）
        self.profile_service = self.memory_manager.profile
        self.preference_service = self.memory_manager.preference
        self.relationship_service = self.memory_manager.relationship
        self.semantic_service = self.memory_manager.semantic
        self.episodic_service = self.memory_manager.episodic
        self.emotion_service = self.memory_manager.emotion

        # =====================
        # LLM
        # =====================
        self.llm_client = DeepSeekClient()

        # =====================
        # Response
        # =====================
        self.response_service = (
            ResponseService(
                self.llm_client
            )
        )

        # =====================
        # Context
        # =====================
        self.context_builder = (
            ContextBuilder()
        )

        self.message_builder = (
            MessageBuilder()
        )

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


# 全局单例
container = ServiceContainer()
