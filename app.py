"""主程序入口。

支持两种输入模式：
- 语音模式（默认）：麦克风 → VAD 分段 → SenseVoice ASR → Agent 流式回复 → TTS 播放
- 文本模式（降级）：语音依赖不可用或模型加载失败时自动回退

架构要点：
    语音模式下，VoicePipeline 作为独立 asyncio task 在后台持续运行（producer），
    识别结果放入队列；主循环从队列取出并在线程池中运行 Agent（consumer）。
    这样 VAD 在 Agent 处理期间不暂停——回声抑制和语音打断持续生效。
"""

import asyncio

import config  # noqa: F401  触发 load_dotenv()，确保 MODELSCOPE_CACHE 等环境变量就绪
from loguru import logger

from core.agent import Agent
from services.container import container

agent = Agent()


async def preload_models() -> bool:
    """启动时预加载 Silero VAD + SenseVoice ASR 模型。

    成功返回 True；语音管线未就绪或加载失败返回 False（降级文本模式）。
    """
    pipeline = container.audio_capture_pipeline
    if pipeline is None:
        logger.warning("[App] 语音管线未就绪（依赖缺失），将使用文本输入模式")
        return False

    try:
        logger.info("[App] 正在预加载 Silero VAD 模型...")
        await asyncio.to_thread(container.vad_service.load)

        logger.info("[App] 正在预加载 SenseVoice ASR 模型（首次加载约 50s）...")
        await asyncio.to_thread(container.sense_voice_asr.load)

        logger.info("[App] 语音模型预加载完成")
        return True
    except Exception as e:  # noqa: BLE001
        logger.error(f"[App] 语音模型预加载失败，将使用文本输入模式: {e}")
        return False


async def _voice_producer(pipeline, utterance_queue: asyncio.Queue):
    """后台任务：持续运行语音管线，将识别结果放入队列。

    作为独立 asyncio task 运行，确保 VAD 在 Agent 处理期间持续工作
    （回声抑制 / 语音打断）。pipeline.stream() 内部 finally 会停止麦克风。
    管线异常（如麦克风不可用）时放入 None 哨兵唤醒 consumer，避免死等。
    """
    try:
        async for voice_input in pipeline.stream():
            await utterance_queue.put(voice_input)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(f"[Voice] 语音管线异常，语音模式将停止: {e}")
        await utterance_queue.put(None)


def _run_agent(user_id: str, text: str):
    """同步执行 Agent 流式对话（在线程池中调用，不阻塞事件循环）。"""
    for _ in agent.stream_chat(user_id, text):
        pass


async def run_voice_mode(user_id: str):
    """语音输入主循环。

    producer 后台运行 VoicePipeline（VAD + ASR），
    consumer 从队列取出识别结果送入 Agent。
    """
    pipeline = container.audio_capture_pipeline
    utterance_queue: asyncio.Queue = asyncio.Queue()

    producer = asyncio.create_task(
        _voice_producer(pipeline, utterance_queue)
    )

    logger.info("[App] 语音模式已启动，请说话...")

    try:
        while True:
            voice_input = await utterance_queue.get()

            # 哨兵：语音管线异常（如麦克风不可用），退出语音模式
            if voice_input is None:
                logger.error("[App] 语音管线已停止")
                break

            logger.info(
                f"[Voice] 识别: {voice_input.text!r} | "
                f"lang={voice_input.language.value} "
                f"emo={voice_input.emotion.value} "
                f"conf={voice_input.confidence:.2f} "
                f"dur={voice_input.duration:.1f}s"
            )

            # 在线程池中运行 Agent，避免阻塞事件循环
            # （VAD 需持续运行以实现回声抑制和语音打断）
            await asyncio.to_thread(
                _run_agent, user_id, voice_input.text
            )
    except asyncio.CancelledError:
        pass
    finally:
        producer.cancel()
        try:
            await producer
        except asyncio.CancelledError:
            pass


def run_text_mode(user_id: str):
    """文本输入模式（语音依赖不可用时的降级）。"""
    logger.info("[App] 文本模式已启动")

    while True:
        try:
            text = input("输入: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue

        for _ in agent.stream_chat(user_id, text):
            pass


async def main():
    logger.info("系统启动开始")

    user_id = "user_001"

    # 启动 TTS Worker（守护线程，进程退出时自动结束）
    container.tts_worker.start()

    # 预加载语音模型
    voice_ready = await preload_models()

    if voice_ready:
        await run_voice_mode(user_id)
    else:
        run_text_mode(user_id)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("系统退出")
