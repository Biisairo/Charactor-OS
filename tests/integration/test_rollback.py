"""Stage 3 후처리 실패 시 상태 롤백 검증 (TASK-02 / REQ-02-1 ~ 02-3).

후처리는 감정·기억·히스토리를 순차로 갱신한다. 중간에 실패하면 이번 턴의
변경이 하나도 남아서는 안 된다. 응답은 이미 사용자에게 전달된 뒤이므로,
상태만 오염된 채 남는 것이 이 경로의 실패 양상이다.

검증 전략:
    감정 갱신은 LLM 호출 **전에** decay와 트리거를 적용하므로, 감정 단계에서
    실패시키면 이미 감정 상태가 변한 상태에서 예외가 발생한다. 롤백이 없으면
    그 변경이 그대로 남는다 — 이 테스트들이 실제로 롤백을 검증한다는 근거다.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import (
    EMOTION_PROMPT_MARKER,
    MEMORY_PROMPT_MARKER,
    PipelineMockClient,
    make_character_os,
)

# 페르소나에 정의된 감정 트리거 키워드 ("아버지" → 분노 0.6, 슬픔 0.4)
TRIGGER_INPUT = "아버지 이야기를 듣고 싶소"


def _seed_state(cos, client) -> None:
    """성공하는 턴을 한 번 실행하여 비어 있지 않은 기준 상태를 만든다."""
    client.next_response.content = "그리 말씀하시니 반갑소."
    cos.chat("처음 뵙겠소")


# ---------------------------------------------------------------------------
# REQ-02-1 — 감정 갱신 실패 시 감정·기억·히스토리가 모두 복원된다
# ---------------------------------------------------------------------------


class TestEmotionStageFailure:
    def test_emotion_state_restored(self, character_dir: Path, tmp_path: Path):
        client = PipelineMockClient()
        cos = make_character_os(character_dir, tmp_path, client)
        _seed_state(cos, client)

        before = dict(cos.emotion.get_state())

        # 이후 감정 갱신만 실패시킨다
        client._fail_when = lambda prompt: EMOTION_PROMPT_MARKER in prompt
        result = cos.chat(TRIGGER_INPUT)

        assert result is None, "후처리 실패 시 chat()은 None을 반환해야 한다"
        assert cos.emotion.get_state() == before, (
            "감정 트리거가 이미 적용된 뒤 실패했으므로 롤백으로 복원되어야 한다"
        )

    def test_history_restored(self, character_dir: Path, tmp_path: Path):
        client = PipelineMockClient()
        cos = make_character_os(character_dir, tmp_path, client)
        _seed_state(cos, client)

        before = cos.history.count()

        client._fail_when = lambda prompt: EMOTION_PROMPT_MARKER in prompt
        cos.chat(TRIGGER_INPUT)

        assert cos.history.count() == before

    def test_no_memory_leaks_through(self, character_dir: Path, tmp_path: Path):
        """기억은 감정 이후 단계이므로 애초에 실행되지 않아야 한다.

        롤백이 아니라 **단계 순서**를 고정하는 테스트다. 기억 갱신이 감정보다
        앞서도록 순서가 바뀌면 이 테스트가 깨진다.
        """
        client = PipelineMockClient()
        cos = make_character_os(character_dir, tmp_path, client)
        _seed_state(cos, client)

        before = cos.memory.snapshot_count()

        client._fail_when = lambda prompt: EMOTION_PROMPT_MARKER in prompt
        cos.chat(TRIGGER_INPUT)

        assert cos.memory.snapshot_count() == before


# ---------------------------------------------------------------------------
# REQ-02-2 — 기억 갱신 실패 시 앞서 완료된 감정·히스토리 갱신도 되돌려진다
# ---------------------------------------------------------------------------


class TestMemoryStageFailure:
    def test_completed_emotion_update_is_reverted(self, character_dir: Path, tmp_path: Path):
        """기억 단계에서 실패해도, 이미 성공한 감정 갱신이 남아서는 안 된다."""
        client = PipelineMockClient()
        cos = make_character_os(character_dir, tmp_path, client)
        _seed_state(cos, client)

        before = dict(cos.emotion.get_state())

        client._fail_when = lambda prompt: MEMORY_PROMPT_MARKER in prompt
        result = cos.chat(TRIGGER_INPUT)

        assert result is None
        assert cos.emotion.get_state() == before

    def test_completed_history_add_is_reverted(self, character_dir: Path, tmp_path: Path):
        client = PipelineMockClient()
        cos = make_character_os(character_dir, tmp_path, client)
        _seed_state(cos, client)

        before = cos.history.count()

        client._fail_when = lambda prompt: MEMORY_PROMPT_MARKER in prompt
        cos.chat(TRIGGER_INPUT)

        assert cos.history.count() == before

    def test_emotion_stage_actually_ran_before_failure(self, character_dir: Path, tmp_path: Path):
        """전제 확인 — 기억 단계 실패 시 감정 단계는 이미 실행된 상태여야 한다.

        이 전제가 깨지면 위 두 테스트는 롤백이 아니라 '아무 일도 없었음'을
        검증하게 되므로, 전제 자체를 명시적으로 고정한다.
        """
        client = PipelineMockClient()
        cos = make_character_os(character_dir, tmp_path, client)
        _seed_state(cos, client)

        client._fail_when = lambda prompt: MEMORY_PROMPT_MARKER in prompt
        cos.chat(TRIGGER_INPUT)

        emotion_calls = [
            record
            for record in client.all_call_records
            if EMOTION_PROMPT_MARKER
            in " ".join(str(m.get("content", "")) for m in record["messages"])
        ]
        assert emotion_calls, "감정 갱신 LLM 호출이 실제로 발생해야 한다"


# ---------------------------------------------------------------------------
# REQ-02-3 — 실패한 턴이 디스크에 부분 저장되지 않는다
# ---------------------------------------------------------------------------


class TestNoPartialPersistence:
    def test_state_files_unchanged_after_failure(self, character_dir: Path, tmp_path: Path):
        """영속화는 모든 갱신이 끝난 뒤에만 일어나므로, 실패 턴은 디스크에 흔적이 없다."""
        client = PipelineMockClient()
        cos = make_character_os(character_dir, tmp_path, client)
        _seed_state(cos, client)

        emotion_file = tmp_path / "emotions.json"
        history_file = tmp_path / "history.json"
        assert emotion_file.exists() and history_file.exists(), (
            "성공한 턴 이후에는 상태 파일이 존재해야 한다"
        )

        emotion_before = emotion_file.read_text(encoding="utf-8")
        history_before = history_file.read_text(encoding="utf-8")

        client._fail_when = lambda prompt: MEMORY_PROMPT_MARKER in prompt
        cos.chat(TRIGGER_INPUT)

        assert emotion_file.read_text(encoding="utf-8") == emotion_before
        assert history_file.read_text(encoding="utf-8") == history_before

    def test_reloaded_state_matches_pre_failure(self, character_dir: Path, tmp_path: Path):
        """디스크에서 다시 읽어도 실패 턴의 내용이 없어야 한다."""
        client = PipelineMockClient()
        cos = make_character_os(character_dir, tmp_path, client)
        _seed_state(cos, client)

        history_file = tmp_path / "history.json"
        turns_before = len(json.loads(history_file.read_text(encoding="utf-8"))["turns"])

        client._fail_when = lambda prompt: MEMORY_PROMPT_MARKER in prompt
        cos.chat(TRIGGER_INPUT)

        fresh_client = PipelineMockClient()
        reloaded = make_character_os(character_dir, tmp_path, fresh_client)
        assert reloaded.history.count() == turns_before
        assert TRIGGER_INPUT not in json.dumps(
            [t.content for t in reloaded.history.get_recent(100)], ensure_ascii=False
        )


# ---------------------------------------------------------------------------
# 대조군 — 성공한 턴은 상태가 남아야 한다
# ---------------------------------------------------------------------------


class TestSuccessfulTurnPersists:
    def test_history_grows_on_success(self, character_dir: Path, tmp_path: Path):
        """롤백 테스트가 '항상 상태가 안 변한다'를 검증하는 게 아님을 보인다."""
        client = PipelineMockClient()
        cos = make_character_os(character_dir, tmp_path, client)

        before = cos.history.count()
        cos.chat("반갑소")

        assert cos.history.count() > before
        assert (tmp_path / "history.json").exists()
