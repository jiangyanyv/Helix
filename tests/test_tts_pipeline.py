"""
端到端测试：验证 StreamChunk -> AudioQueue -> TTSWorker 链路通畅。

核心验证点：
1. Turn 正确创建后，chunk.turn_id 能通过 is_tts_playable 检查（不会被丢弃）
2. response_generator_node 放入 container.audio_queue 的 chunk，
   TTSWorker 能真正消费到（同一个 Queue 实例）
3. 流式 Chunk 能从 Agent.stream_chat 逐个 yield 出来

不依赖真实 LLM API：用 Mock 替换 DeepSeekClient.stream。
"""

import time
from unittest.mock import patch, MagicMock

from services.container import container
from core.agent import Agent


def build_token_stream():
    """模拟LLM返回的token序列，故意包含标点让StreamProcessor切成多句"""
    tokens = [
        "你好呀，",
        "主人！",
        "今天想聊些什么呢？",
        "要不要说说",
        "最近有趣的事情？",
    ]
    for t in tokens:
        yield t


def test_tts_pipeline_end_to_end():
    print()
    print("=" * 60)
    print("  TEST: TTS 链路端到端贯通验证")
    print("=" * 60)

    # =========================================================
    # 0. 准备：启动 TTS Worker，清空调试记录
    # =========================================================
    tts = container.tts_worker
    audio_queue = container.audio_queue
    runtime = container.runtime_manager

    if not tts.is_running():
        tts.start()

    tts.played_chunks.clear()
    audio_queue.clear()

    # =========================================================
    # 1. Mock LLM Client：绕过真实 API 调用
    # =========================================================
    mock_llm = MagicMock()
    mock_llm.stream = lambda request: build_token_stream()

    with patch.object(
        container.response_service, "llm", mock_llm
    ):
        agent = Agent()
        session_id = "test_user_001"

        # =========================================================
        # 2. 发起一次 stream_chat，收集流式输出
        # =========================================================
        print()
        print("🤖 调用 Agent.stream_chat ...")

        streamed_tokens = []
        for token in agent.stream_chat(session_id, "你好"):
            streamed_tokens.append(token)
            print(f"  ↳ yield: {token!r}")

        full_output = "".join(streamed_tokens)
        print()
        print(f"📝 流式输出总内容: {full_output!r}")

    # =========================================================
    # 3. 等待 TTS Worker 消费 Queue（TTS 播放每段模拟 1 秒）
    # =========================================================
    print()
    print("⏳ 等待 TTS Worker 消费队列 ...")

    # 最多等 6 秒
    waited = 0.0
    while waited < 6.0:
        if audio_queue.empty() and len(tts.played_chunks) >= 1:
            break
        time.sleep(0.3)
        waited += 0.3

    time.sleep(1.5)  # 给最后一段 Chunk 模拟播放的时间

    # =========================================================
    # 4. 断言验证
    # =========================================================
    print()
    print("=" * 60)
    print("  RESULT")
    print("=" * 60)

    print(f"Agent 流式输出片段数: {len(streamed_tokens)}")
    print(f"TTS 实际播放 Chunk 数: {len(tts.played_chunks)}")
    for i, c in enumerate(tts.played_chunks, 1):
        print(f"  TTS #{i}: turn={c.turn_id[:8]}... text={c.text!r}")

    # ----------- 关键断言 1: Agent 流式输出不为空 -----------
    assert len(streamed_tokens) >= 1, (
        "❌ Agent.stream_chat 没有 yield 任何 token 出来"
    )

    # ----------- 关键断言 2: TTS 至少播放了一个 Chunk -----------
    assert len(tts.played_chunks) >= 1, (
        "❌ TTSWorker 一个 Chunk 都没播放！"
        "大概率是：Queue 实例不一致 / turn_id 没传导致被丢弃"
    )

    # ----------- 关键断言 3: 每个播放的 Chunk 都有有效的 turn_id -----------
    for c in tts.played_chunks:
        assert c.turn_id is not None and len(c.turn_id) > 0, (
            f"❌ Chunk 的 turn_id 为空：{c}"
        )

    # ----------- 关键断言 4: 内容一致性 -----------
    tts_total_text = "".join(c.text for c in tts.played_chunks)
    assert tts_total_text == full_output, (
        f"❌ TTS 播放内容与 Agent 输出不一致\n"
        f"   Agent: {full_output!r}\n"
        f"   TTS  : {tts_total_text!r}"
    )

    print()
    print("=" * 60)
    print("✅ TTS PIPELINE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_tts_pipeline_end_to_end()
