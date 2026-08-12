"""LLM 상호작용 층의 파싱과 실패 처리 (TASK-14, REQ-14-3).

프롬프트 조립·LLM 호출·응답 파싱을 도메인 모듈에서 떼어낸 뒤, **응답을 해석하는
책임은 전부 이 층에 있다.** 분리 전에는 같은 파싱·거부 처리가 세 곳(기억 추출,
충돌 판정, 감정 분석)에 흩어져 있었고, TASK-12의 거부 처리도 세 곳에 각각 심어야 했다.

이 층의 계약은 하나다 — **답을 해석할 수 없으면 "아무 제안 없음"을 돌려준다.**
예외를 던져 후처리를 중단시키지 않는다. 응답 생성은 이미 끝났고, 상태 갱신 실패로
대화를 무르는 것은 과하다.

LLM API 호출을 하지 않는다.
"""

from __future__ import annotations

import json

from src.analysis import (
    ConflictClassifier,
    EmotionAnalyzer,
    MemoryCandidate,
    MemoryExtractor,
)
from src.llm.client import TrimmedMessage

REFUSAL = "The request was rejected because it was considered high risk"


class _Client:
    """정해진 문자열을 돌려주는 최소 클라이언트."""

    def __init__(self, content: str):
        self._content = content
        self.prompts: list[str] = []

    def call_llm(self, messages, tools=None, use_stream=False, mute=True, **kwargs):
        self.prompts.append(" ".join(str(m.get("content", "")) for m in messages))
        return TrimmedMessage(
            content=self._content, role="assistant", reasoning_content="", tool_calls=[], usage=None
        )


def _json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# MemoryExtractor
# ---------------------------------------------------------------------------


class TestMemoryExtractor:
    def test_parses_candidates(self):
        client = _Client(_json({"memories": [{"content": "시험이 있다", "importance": 0.8}]}))

        result = MemoryExtractor(client).extract("입력", "응답")

        assert result == [MemoryCandidate("시험이 있다", 0.8)]

    def test_applies_default_importance(self):
        client = _Client(_json({"memories": [{"content": "내용"}]}))

        assert MemoryExtractor(client).extract("입력", "응답")[0].importance == 0.5

    def test_skips_empty_content(self):
        client = _Client(_json({"memories": [{"content": ""}, {"content": "실제"}]}))

        result = MemoryExtractor(client).extract("입력", "응답")

        assert [c.content for c in result] == ["실제"]

    def test_refusal_yields_no_candidates(self):
        assert MemoryExtractor(_Client(REFUSAL)).extract("입력", "응답") == []

    def test_malformed_json_yields_no_candidates(self):
        assert MemoryExtractor(_Client("JSON 아님")).extract("입력", "응답") == []

    def test_missing_key_yields_no_candidates(self):
        assert MemoryExtractor(_Client(_json({}))).extract("입력", "응답") == []

    def test_prompt_hook_receives_prompt(self):
        seen: list[tuple[str, str]] = []
        client = _Client(_json({"memories": []}))

        MemoryExtractor(client, on_prompt=lambda m, p: seen.append((m, p))).extract("입력", "응답")

        assert [m for m, _ in seen] == ["memory"]
        assert "입력" in seen[0][1]

    def test_history_context_reaches_prompt(self):
        client = _Client(_json({"memories": []}))

        MemoryExtractor(client).extract("입력", "응답", history_context="[최근 대화] 지난 얘기")

        assert "지난 얘기" in client.prompts[0]


# ---------------------------------------------------------------------------
# ConflictClassifier
# ---------------------------------------------------------------------------


class TestConflictClassifier:
    def test_parses_classification(self):
        client = _Client(_json({"classification": "SIMILAR"}))

        assert ConflictClassifier(client).classify("기존", "신규") == "SIMILAR"

    def test_refusal_falls_back_to_different(self):
        assert ConflictClassifier(_Client(REFUSAL)).classify("기존", "신규") == "DIFFERENT"

    def test_malformed_json_falls_back_to_different(self):
        assert ConflictClassifier(_Client("JSON 아님")).classify("기존", "신규") == "DIFFERENT"

    def test_unknown_label_falls_back_to_different(self):
        """모르는 값을 그대로 통과시키면 도메인이 예상 못 한 분기를 탄다."""
        client = _Client(_json({"classification": "MAYBE"}))

        assert ConflictClassifier(client).classify("기존", "신규") == "DIFFERENT"

    def test_both_memories_reach_prompt(self):
        client = _Client(_json({"classification": "IDENTICAL"}))

        ConflictClassifier(client).classify("기존 기억 내용", "새 기억 내용")

        assert "기존 기억 내용" in client.prompts[0]
        assert "새 기억 내용" in client.prompts[0]


# ---------------------------------------------------------------------------
# EmotionAnalyzer
# ---------------------------------------------------------------------------


class TestEmotionAnalyzer:
    def test_parses_significant_change(self):
        client = _Client(
            _json({"significant": True, "emotions": {"기쁨": 0.6}, "remove": ["분노"]})
        )

        result = EmotionAnalyzer(client).analyze("입력", "응답", {})

        assert result.significant is True
        assert result.emotions == {"기쁨": 0.6}
        assert result.remove == ["분노"]

    def test_insignificant_change_carries_no_payload(self):
        client = _Client(_json({"significant": False, "emotions": {"기쁨": 0.9}}))

        result = EmotionAnalyzer(client).analyze("입력", "응답", {})

        assert result.significant is False
        assert result.emotions == {}

    def test_refusal_yields_no_change(self):
        assert EmotionAnalyzer(_Client(REFUSAL)).analyze("입력", "응답", {}).significant is False

    def test_malformed_json_yields_no_change(self):
        assert (
            EmotionAnalyzer(_Client("JSON 아님")).analyze("입력", "응답", {}).significant is False
        )

    def test_null_fields_become_empty(self):
        client = _Client(_json({"significant": True, "emotions": None, "remove": None}))

        result = EmotionAnalyzer(client).analyze("입력", "응답", {})

        assert result.emotions == {}
        assert result.remove == []

    def test_current_state_reaches_prompt(self):
        client = _Client(_json({"significant": False}))

        EmotionAnalyzer(client).analyze("입력", "응답", {"연민": 0.42})

        assert "연민" in client.prompts[0]

    def test_empty_state_is_rendered_as_empty_object(self):
        client = _Client(_json({"significant": False}))

        EmotionAnalyzer(client).analyze("입력", "응답", {})

        assert "{}" in client.prompts[0]
