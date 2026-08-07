from services.llm.deepseek_client import (
    DeepSeekClient
)

from services.response.response_service import (
    ResponseService
)

from services.context.context_builder import (
    ContextBuilder
)

from services.context.message_builder import (
    MessageBuilder
)
from services.memory.memory_manager import (
    MemoryManager
)
from services.event.event_bus import EventBus
from voice.queue.audio_queue import (
    AudioQueue
)

class ServiceContainer:
    """
    全局服务容器

    管理所有单例Service
    """


    def __init__(self):

        # =====================
        # Memory
        # =====================
        self.memory_manager = (
            MemoryManager()
        )

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


        self.event_bus = EventBus()

        self.audio_queue = AudioQueue()


# 全局实例

container = ServiceContainer()