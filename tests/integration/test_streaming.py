"""chat_stream() 스트리밍 파이프라인 검증 (TASK-02 / REQ-02-4, REQ-02-5).

chat_stream은 Stage 2를 별도 스레드에서 실행하고 토큰을 큐로 넘겨 받는다.
제너레이터가 소비되는 동안 토큰이 순차로 나오고, 스트림이 끝난 뒤에
후처리가 정확히 한 번 실행되어야 한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import (
    EMOTION_PROMPT_MARKER,
    PipelineMockClient,
    make_character_os,
)

RESPONSE = "그대의 뜻을 알겠소"


# ---------------------------------------------------------------------------
# REQ-02-4 — 토큰이 순차로 yield되고, 스트림 종료 후 후처리가 1회 실행된다
# ---------------------------------------------------------------------------


class TestStreamingTokens:
    def test_yields_tokens(self, character_dir: Path, tmp_path: Path):
        client = PipelineMockClient(response=RESPONSE)
        cos = make_character_os(character_dir, tmp_path, client)

        tokens = list(cos.chat_stream("안녕하시오"))

        assert len(tokens) > 1, "응답이 여러 토큰으로 나뉘어 전달되어야 한다"
        assert "".join(tokens) == RESPONSE

    def test_tokens_arrive_in_order(self, character_dir: Path, tmp_path: Path):
        """MockClient는 문자 단위로 콜백하므로 순서가 그대로 보존되어야 한다."""
        client = PipelineMockClient(response=RESPONSE)
        cos = make_character_os(character_dir, tmp_path, client)

        tokens = list(cos.chat_stream("안녕하시오"))

        assert tokens == list(RESPONSE)

    def test_post_processing_runs_after_stream(self, character_dir: Path, tmp_path: Path):
        """스트림을 모두 소비하면 이번 턴이 히스토리에 반영된다."""
        client = PipelineMockClient(response=RESPONSE)
        cos = make_character_os(character_dir, tmp_path, client)

        before = cos.history.count()
        list(cos.chat_stream("안녕하시오"))

        assert cos.history.count() == before + 2  # user + character

    def test_post_processing_runs_once(self, character_dir: Path, tmp_path: Path):
        """감정 갱신 LLM 호출이 정확히 1회여야 한다 (후처리 중복 실행 방지)."""
        client = PipelineMockClient(response=RESPONSE)
        cos = make_character_os(character_dir, tmp_path, client)

        list(cos.chat_stream("안녕하시오"))

        emotion_calls = [
            record
            for record in client.all_call_records
            if EMOTION_PROMPT_MARKER
            in " ".join(str(m.get("content", "")) for m in record["messages"])
        ]
        assert len(emotion_calls) == 1

    def test_stream_requested_from_client(self, character_dir: Path, tmp_path: Path):
        """Stage 2는 스트리밍 모드로 LLM을 호출해야 한다."""
        client = PipelineMockClient(response=RESPONSE)
        cos = make_character_os(character_dir, tmp_path, client)

        list(cos.chat_stream("안녕하시오"))

        assert any(record["use_stream"] for record in client.all_call_records)

    def test_not_consumed_means_no_side_effect(self, character_dir: Path, tmp_path: Path):
        """제너레이터를 소비하지 않으면 파이프라인이 실행되지 않는다."""
        client = PipelineMockClient(response=RESPONSE)
        cos = make_character_os(character_dir, tmp_path, client)

        before = cos.history.count()
        cos.chat_stream("안녕하시오")  # 소비하지 않음

        assert cos.history.count() == before
        assert client.call_count == 0


# ---------------------------------------------------------------------------
# REQ-02-5 — 스트리밍 중 LLM 실패 시 예외가 전달되고 상태가 오염되지 않는다
# ---------------------------------------------------------------------------


class TestStreamingFailure:
    def test_exception_propagates_to_consumer(self, character_dir: Path, tmp_path: Path):
        """워커 스레드에서 발생한 예외가 소비자에게 전달되어야 한다.

        큐로 스레드 경계를 넘기 때문에, 예외를 명시적으로 전달하지 않으면
        소비자는 빈 스트림을 정상 종료로 오인한다.
        """
        client = PipelineMockClient(response=RESPONSE, fail_when=lambda _p: True)
        cos = make_character_os(character_dir, tmp_path, client)

        with pytest.raises(RuntimeError, match="주입된 LLM 실패"):
            list(cos.chat_stream("안녕하시오"))

    def test_state_unchanged_on_failure(self, character_dir: Path, tmp_path: Path):
        client = PipelineMockClient(response=RESPONSE, fail_when=lambda _p: True)
        cos = make_character_os(character_dir, tmp_path, client)

        history_before = cos.history.count()
        emotion_before = dict(cos.emotion.get_state())
        memory_before = cos.memory.snapshot_count()

        with pytest.raises(RuntimeError):
            list(cos.chat_stream("아버지 이야기를 듣고 싶소"))

        assert cos.history.count() == history_before
        assert cos.emotion.get_state() == emotion_before
        assert cos.memory.snapshot_count() == memory_before

    def test_no_state_file_written_on_failure(self, character_dir: Path, tmp_path: Path):
        client = PipelineMockClient(response=RESPONSE, fail_when=lambda _p: True)
        cos = make_character_os(character_dir, tmp_path, client)

        with pytest.raises(RuntimeError):
            list(cos.chat_stream("안녕하시오"))

        assert not (tmp_path / "history.json").exists()
        assert not (tmp_path / "emotions.json").exists()

    def test_post_processing_failure_does_not_break_stream(
        self, character_dir: Path, tmp_path: Path
    ):
        """후처리가 실패해도 이미 전달된 토큰은 유효하며 예외가 새어 나오지 않는다.

        응답은 사용자에게 이미 도달했으므로, 후처리 실패로 스트림 자체를
        오류 처리하는 것은 옳지 않다.
        """
        client = PipelineMockClient(response=RESPONSE)
        cos = make_character_os(character_dir, tmp_path, client)

        # Stage 2는 통과시키고 후처리(감정 갱신)만 실패시킨다
        client._fail_when = lambda prompt: EMOTION_PROMPT_MARKER in prompt

        tokens = list(cos.chat_stream("아버지 이야기를 듣고 싶소"))

        assert "".join(tokens) == RESPONSE
        assert cos.history.count() == 0, "실패한 후처리는 롤백되어야 한다"
