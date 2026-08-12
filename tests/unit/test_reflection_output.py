"""검토문 출력 축소 (TASK-09 / REQ-09-1, 09-4).

TASK-04 실측에서 검토기가 **턴당 941 출력 토큰**을 썼다. 파이프라인에서 가장 긴
출력이고, Reflection 추가 비용의 절반가량이 응답 개선이 아니라 검토문 작성에
쓰이고 있었다. 프롬프트가 형식을 강제하지 않아 모델이 7개 기준을 하나씩
논평했기 때문이다.

파싱을 순수 함수로 분리해 LLM 없이 검증한다.
"""

from __future__ import annotations

import json

import pytest

from src.llm.client import TrimmedMessage
from src.modules.reflection import (
    MAX_FEEDBACK_CHARS,
    ReflectionReviewer,
    parse_review_response,
)


class _StubPersona:
    _data = {
        "name": "홍길동",
        "speaking_style": {"summary": "고어체", "tone": "차분함"},
        "behavior": {"rules": ["거짓말을 하지 않는다"]},
    }


class _StubEmotion:
    def get_state(self):
        return {"분노": 0.6}


class _RecordingClient:
    """호출 인자를 잡아두고 정해진 응답을 돌려준다."""

    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict] = []

    def call_llm(self, messages, tools=None, use_stream=False, mute=True, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return TrimmedMessage(
            content=self._content, role="assistant", reasoning_content="", tool_calls=[], usage=None
        )


def _reviewer(content: str) -> tuple[ReflectionReviewer, _RecordingClient]:
    client = _RecordingClient(content)
    return ReflectionReviewer(client, _StubPersona(), _StubEmotion()), client


# ---------------------------------------------------------------------------
# REQ-09-1 — 형식이 강제된다
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    def test_review_requests_json_object(self):
        reviewer, client = _reviewer('{"verdict": "PASS", "feedback": ""}')

        reviewer.review("안녕", "반갑소")

        assert client.calls[0]["response_format"] == {"type": "json_object"}

    def test_prompt_specifies_the_json_shape_and_a_length_cap(self):
        reviewer, _ = _reviewer('{"verdict": "PASS"}')

        prompt = reviewer._build_review_prompt("안녕", "반갑소")

        assert '"verdict"' in prompt
        assert '"feedback"' in prompt
        assert "80자" in prompt
        assert "기준을 하나씩 논평하지 마세요" in prompt

    @pytest.mark.parametrize("verdict", ["PASS", "pass", "Pass"])
    def test_pass_is_approved(self, verdict):
        result = parse_review_response(json.dumps({"verdict": verdict, "feedback": ""}))

        assert result.approved

    def test_fail_carries_feedback(self):
        result = parse_review_response(
            json.dumps({"verdict": "FAIL", "feedback": "기준 5 위반 — '서울'을 '한양'으로"})
        )

        assert not result.approved
        assert "한양" in result.feedback


# ---------------------------------------------------------------------------
# 폴백 — 프로바이더가 형식을 무시해도 검토가 뒤집히지 않는다
# ---------------------------------------------------------------------------


class TestLegacyTextFallback:
    def test_plain_pass_still_approves(self):
        """폴백이 없으면 검토가 전부 FAIL로 뒤집혀 매 턴 재생성이 돈다."""
        assert parse_review_response("PASS").approved

    def test_plain_fail_extracts_reason(self):
        result = parse_review_response("FAIL: 기준 4 위반 — 중국어로 답했습니다")

        assert not result.approved
        assert result.feedback.startswith("기준 4 위반")

    def test_unparseable_text_is_treated_as_failure_with_the_text_as_feedback(self):
        result = parse_review_response("음... 이 응답은 조금 애매합니다")

        assert not result.approved
        assert "애매" in result.feedback

    def test_empty_response_is_not_silently_approved(self):
        result = parse_review_response("")

        assert not result.approved
        assert result.feedback

    def test_json_without_verdict_falls_back_to_text(self):
        """형식은 JSON인데 필드가 없으면 텍스트로 읽는다."""
        result = parse_review_response('{"comment": "좋습니다"}')

        assert not result.approved


# ---------------------------------------------------------------------------
# REQ-09-4 — 재생성에 넘어가는 피드백이 방향을 담는다
# ---------------------------------------------------------------------------


class TestFeedbackReachesRegeneration:
    def test_feedback_is_passed_to_regenerate(self):
        reviewer, _ = _reviewer(
            json.dumps({"verdict": "FAIL", "feedback": "기준 6 위반 — 코드를 지우세요"})
        )
        received: list[str] = []

        reviewer.review_and_improve(
            "코드 짜줘", "def f(): ...", lambda fb: received.append(fb) or "고침"
        )

        assert received and "코드를 지우세요" in received[0]

    def test_long_feedback_is_capped(self):
        """모델이 지시를 어겨도 재생성 프롬프트가 비대해지지 않는다."""
        result = parse_review_response(json.dumps({"verdict": "FAIL", "feedback": "가" * 500}))

        assert len(result.feedback) <= MAX_FEEDBACK_CHARS
