
import threading
import time

from core.runtime.runtime_manager import RuntimeManager

from services.llm.stream_chunk import StreamChunk

from voice.queue.audio_queue import AudioQueue
from voice.tts.tts_worker import TTSWorker


def test_interruptible_tts():

    print()
    print("=" * 60)
    print("      TEST: interruptible=True")
    print("=" * 60)

    # =========================================================
    # 1. Runtime
    # =========================================================

    runtime = RuntimeManager()

    # =========================================================
    # 2. Audio Queue
    # =========================================================

    audio_queue = AudioQueue()

    # =========================================================
    # 3. 创建 Turn
    # =========================================================

    turn = runtime.start_turn(
        user_id="interrupt_test"
    )

    turn_id = turn.turn_id

    print(
        f"🟢 Turn created: {turn_id}"
    )

    # =========================================================
    # 4. TTS Worker
    # =========================================================

    tts_worker = TTSWorker(
        audio_queue=audio_queue,
        runtime_manager=runtime
    )

    tts_worker.start()

    time.sleep(0.2)

    # =========================================================
    # 5. 创建一个可以被打断的 Chunk
    # =========================================================

    chunk = StreamChunk(
        text="这是一段正在播放中的语音，现在应该被用户打断。",
        turn_id=turn_id,
        is_sentence_end=True,
        interruptible=True
    )

    print()
    print(
        "📦 放入 interruptible=True Chunk"
    )

    audio_queue.put(chunk)

    # =========================================================
    # 6. 等待 TTS 开始播放
    # =========================================================

    time.sleep(0.3)

    print()
    print(
        "🔊 当前 TTS 应该正在播放..."
    )

    print(
        f"TTS playing: "
        f"{runtime.state.tts_playing}"
    )

    # =========================================================
    # 7. 用户打断
    # =========================================================

    print()
    print("=" * 60)
    print("🎤 USER INTERRUPT")
    print("=" * 60)

    # 先让当前 Turn 失效
    runtime.interrupt(
        turn_id
    )

    # 再停止 TTS
    tts_worker.stop()

    # =========================================================
    # 8. 等待 TTS 响应
    # =========================================================

    time.sleep(0.3)

    # =========================================================
    # 9. 检查结果
    # =========================================================

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        f"Turn accepting LLM: "
        f"{runtime.is_llm_accepting(turn_id)}"
    )

    print(
        f"TTS playing: "
        f"{runtime.state.tts_playing}"
    )

    # =========================================================
    # 10. 断言
    # =========================================================

    assert not runtime.is_llm_accepting(
        turn_id
    ), (
        "Turn should not be accepting LLM "
        "after interrupt."
    )

    assert not runtime.state.tts_playing, (
        "TTS should stop after "
        "interrupting an interruptible chunk."
    )

    print()
    print("=" * 60)
    print("✅ interruptible=True TEST PASSED")
    print("=" * 60)



def test_non_interruptible_tts():

    print()
    print("=" * 60)
    print("      TEST: interruptible=False")
    print("=" * 60)

    # =========================================================
    # 1. Runtime
    # =========================================================

    runtime = RuntimeManager()

    # =========================================================
    # 2. Audio Queue
    # =========================================================

    audio_queue = AudioQueue()

    # =========================================================
    # 3. 创建 Turn
    # =========================================================

    turn = runtime.start_turn(
        user_id="non_interruptible_test"
    )

    turn_id = turn.turn_id

    print(
        f"🟢 Turn created: {turn_id}"
    )

    # =========================================================
    # 4. TTS Worker
    # =========================================================

    tts_worker = TTSWorker(
        audio_queue=audio_queue,
        runtime_manager=runtime
    )

    tts_worker.start()

    time.sleep(0.2)

    # =========================================================
    # 5. 第一个 Chunk
    #
    # interruptible=False
    #
    # 当前 Chunk 即使用户打断，
    # 也应该继续播放完成。
    # =========================================================

    chunk_1 = StreamChunk(
        text="这是一个不能被中途打断的Chunk。",
        turn_id=turn_id,
        is_sentence_end=True,
        interruptible=False
    )

    print()
    print(
        "📦 放入 Chunk 1 | interruptible=False"
    )

    audio_queue.put(
        chunk_1
    )

    # =========================================================
    # 6. 等待 Chunk 1 开始播放
    # =========================================================

    time.sleep(0.3)

    print()
    print(
        "🔊 Chunk 1 应该正在播放..."
    )

    print(
        f"TTS playing: "
        f"{runtime.state.tts_playing}"
    )

    # =========================================================
    # 7. 用户打断
    # =========================================================

    print()
    print("=" * 60)
    print("🎤 USER INTERRUPT")
    print("=" * 60)

    # 让当前 Turn 失效
    runtime.interrupt(
        turn_id
    )

    # 请求停止 TTS
    #
    # 但是因为 Chunk 1 是
    # interruptible=False，
    # 当前 Chunk 应该不会被立即停止。
    tts_worker.stop()

    # =========================================================
    # 8. 在旧 Turn 中继续放入 Chunk 2
    #
    # 这个 Chunk 已经属于一个
    # 被 interrupt 的旧 Turn。
    #
    # 所以无论 interruptible 是什么，
    # 都不能播放。
    # =========================================================

    chunk_2 = StreamChunk(
        text="这是旧Turn后续产生的Chunk，不能再播放。",
        turn_id=turn_id,
        is_sentence_end=True,
        interruptible=True
    )

    print()
    print(
        "📦 放入 Chunk 2 | "
        "旧 Turn | interruptible=True"
    )

    audio_queue.put(
        chunk_2
    )

    # =========================================================
    # 9. 等待 Chunk 1 播放完成
    #
    # 当前模拟 TTS 播放时间为 1 秒。
    # =========================================================

    time.sleep(1.2)

    # =========================================================
    # 10. 等待 Worker 继续消费 Queue
    # =========================================================

    time.sleep(0.3)

    # =========================================================
    # 11. 检查结果
    # =========================================================

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        f"Turn accepting LLM: "
        f"{runtime.is_llm_accepting(turn_id)}"
    )

    print(
        f"Played chunks: "
        f"{len(tts_worker.played_chunks)}"
    )

    for index, played_chunk in enumerate(
            tts_worker.played_chunks,
            start=1
    ):

        print(
            f"Played #{index}: "
            f"turn={played_chunk.turn_id} | "
            f"text={played_chunk.text!r}"
        )

    # =========================================================
    # 12. 核心断言
    # =========================================================

    # Turn 必须已经不再接受 LLM 产出
    assert not runtime.is_llm_accepting(
        turn_id
    ), (
        "Turn should not be accepting LLM "
        "after interrupt."
    )

    # 只能播放第一个 Chunk
    assert len(
        tts_worker.played_chunks
    ) == 1, (
        "Only the current non-interruptible "
        "Chunk should be played."
    )

    # 确认播放的是 Chunk 1
    assert (
        tts_worker.played_chunks[0].text
        == chunk_1.text
    ), (
        "The played Chunk should be Chunk 1."
    )

    # 确认 Chunk 2 没有播放
    assert all(
        played.text != chunk_2.text
        for played in tts_worker.played_chunks
    ), (
        "Old Turn Chunk 2 should NOT be played."
    )

    print()
    print("=" * 60)
    print(
        "✅ interruptible=False TEST PASSED"
    )
    print("=" * 60)

