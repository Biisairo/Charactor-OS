"""프로바이더 거부 응답 차단 (TASK-11, REQ-11-3 · REQ-11-4).

프로바이더가 콘텐츠 필터로 요청을 거부하면 거부 메시지가 캐릭터 응답 자리에
그대로 담겨 온다. 이를 그대로 사용자에게 보여주거나 상태에 저장하면,
인프라 장애가 캐릭터 발화로 위장되고 이후 턴의 프롬프트까지 오염된다.

실제로 웹 UI에서 관측된 결함이다. 히스토리에
"The request was rejected because it was considered high risk"가 2건 남아
이후 턴의 [최근 대화] 컨텍스트로 들어가고 있었다.

이 테스트는 LLM API 키 없이 결정론적으로 통과한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.tools import FINISH_TOOL
from src.llm.client import TrimmedMessage
from tests.conftest import (
    MockClient,
    PipelineMockClient,
    default_brain_script,
    make_character_os,
)

REFUSAL = "The request was rejected because it was considered high risk"
VALID = "흠, 그리 말하니 반갑구나."


class SequenceClient(MockClient):
    """호출 순서대로 정해진 응답을 돌려주는 더블.

    마지막 응답에 도달하면 그 값을 계속 반환한다.
    """

    def __init__(self, responses: list[str], brain_ok: bool = True):
        super().__init__(response=responses[0])
        self._responses = list(responses)
        # 검증 대상은 Stage 2의 거부 처리다. 뇌까지 거부시키면 턴이 그 앞에서
        # 끝나 정작 보려는 경로를 지나지 않는다.
        self._brain_ok = brain_ok
        self._brain_steps: list = []

    def call_llm(
        self,
        messages,
        tools=None,
        use_stream=False,
        mute=True,
        response_format=None,
        token_callback=None,
        max_tokens=None,
    ) -> TrimmedMessage:
        is_brain = bool(tools) and any(
            t.get("function", {}).get("name") == FINISH_TOOL for t in tools
        )
        if is_brain and self._brain_ok:
            if not self._brain_steps:
                self._brain_steps = default_brain_script()
            self.call_count += 1
            return self._brain_steps.pop(0)

        index = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        self.last_messages = messages
        return TrimmedMessage(
            content=self._responses[index],
            role="assistant",
            reasoning_content="",
            tool_calls=[],
            usage=None,
        )


@pytest.fixture
def refusing_cos(character_dir: Path, tmp_path: Path):
    """모든 호출이 거부를 반환하는 CharacterOS."""
    client = PipelineMockClient(response=REFUSAL)
    cos = make_character_os(character_dir, tmp_path / "state", client)
    return cos, client


# ---------------------------------------------------------------------------
# REQ-11-3 — 거부 응답이 사용자에게 표시되지 않는다
# ---------------------------------------------------------------------------


class TestRefusalNotShown:
    def test_refusal_is_not_returned_as_response(self, refusing_cos):
        cos, _ = refusing_cos
        result = cos.chat("안녕")

        assert result != REFUSAL
        assert result is None, "거부가 지속되면 응답이 아니라 실패로 끝나야 한다"

    def test_empty_response_is_treated_as_refusal(self, character_dir: Path, tmp_path: Path):
        client = PipelineMockClient(response="   ")
        cos = make_character_os(character_dir, tmp_path / "state", client)

        assert cos.chat("안녕") is None

    def test_retries_before_giving_up(self, refusing_cos):
        cos, client = refusing_cos
        cos.chat("안녕")

        # 검토가 꺼져 있으므로 Stage 2의 호출은 응답 생성뿐이고,
        # 거부로 끝나면 Stage 3에 진입하지 않는다. 즉 전체 호출 수 = 시도 횟수.
        assert client.call_count > 1, "거부 시 재시도 없이 포기해서는 안 된다"

    def test_valid_response_after_refusal_is_used(self, character_dir: Path, tmp_path: Path):
        client = SequenceClient([REFUSAL, VALID])
        cos = make_character_os(character_dir, tmp_path / "state", client)

        assert cos.chat("안녕") == VALID

    def test_normal_response_is_not_retried(self, character_dir: Path, tmp_path: Path):
        client = PipelineMockClient(response=VALID)
        cos = make_character_os(character_dir, tmp_path / "state", client)
        cos.chat("안녕")

        # 뇌 호출도 시스템 프롬프트에 페르소나를 싣는다. 응답 호출은 [응답 규칙]으로 가른다.
        response_calls = [
            record
            for record in client.all_call_records
            if any(
                m.get("role") == "system" and "[응답 규칙]" in str(m.get("content", ""))
                for m in record["messages"]
            )
        ]
        assert len(response_calls) == 1, "정상 응답에 재시도가 붙어서는 안 된다"


# ---------------------------------------------------------------------------
# REQ-11-4 — 거부 응답이 상태에 저장되지 않는다
# ---------------------------------------------------------------------------


class TestRefusalNotPersisted:
    def test_history_is_not_polluted(self, refusing_cos):
        cos, _ = refusing_cos
        cos.chat("안녕")

        assert cos.history.count() == 0

    def test_memory_is_not_polluted(self, refusing_cos):
        cos, _ = refusing_cos
        cos.chat("안녕")

        assert cos.memory.snapshot_count() == 0

    def test_emotion_is_not_updated(self, refusing_cos):
        cos, _ = refusing_cos
        before = dict(cos.emotion.get_state())
        cos.chat("안녕")

        assert cos.emotion.get_state() == before

    def test_nothing_written_to_disk(self, refusing_cos, tmp_path: Path):
        cos, _ = refusing_cos
        cos.chat("안녕")

        history_file = tmp_path / "state" / "history.json"
        if history_file.exists():
            assert REFUSAL not in history_file.read_text(encoding="utf-8")

    def test_refusal_absent_from_next_prompt(self, character_dir: Path, tmp_path: Path):
        """거부 턴 이후의 프롬프트에 거부 문자열이 섞이지 않아야 한다."""
        client = SequenceClient([REFUSAL, REFUSAL, REFUSAL, VALID])
        cos = make_character_os(character_dir, tmp_path / "state", client)

        cos.chat("첫 턴")
        cos.chat("두 번째 턴")

        assert client.last_messages is not None
        joined = " ".join(str(m.get("content", "")) for m in client.last_messages)
        assert REFUSAL not in joined
