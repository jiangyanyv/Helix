from time import sleep

from core.agent import Agent
from services.container import container
from loguru import logger

agent = Agent()


if __name__ == "__main__":

    logger.info("系统启动开始")

    session_id = "user001"

    # =========================================================
    # 1. 使用 container 中的全局 RuntimeManager + TTSWorker
    #    （确保 TTSWorker 监听的 AudioQueue 与 Graph 写入的 AudioQueue 是同一个实例）
    # =========================================================

    runtime = container.runtime_manager
    audio_queue = container.audio_queue
    tts = container.tts_worker

    # 启动 TTS Worker 线程
    tts.start()

    while True:

        sleep(0.5)
        text = input("输入: ")

        # 清空 TTS 调试记录（可选）
        if hasattr(tts, "played_chunks"):
            tts.played_chunks.clear()

        tokens = []
        for token in agent.stream_chat(
            session_id,
            text
        ):
            pass

            tokens.append(token)
        output = "".join(tokens)
        logger.info(output)

        # 简单验证 TTS 是否收到数据
        sleep(3)
        if hasattr(tts, "played_chunks") and tts.played_chunks:
            logger.info(f"TTS 已播放 {len(tts.played_chunks)} 个 Chunk")
        else:
            logger.warning("TTS 没有播放任何 Chunk，请检查链路")

        # print()
