"""판정 재시도와 지연 측정 (TASK-01 후속).

사례를 버리면 표본이 줄고, 실패가 특정 설정에 몰리면 설정 간 비교가 편향된다.
실제로 판정자가 축 하나를 빠뜨려 사례가 통째로 사라진 적이 있다.
"""

from __future__ import annotations

import pytest

from eval.dataset import GoldenCase
from eval.judge import MAX_JUDGE_ATTEMPTS, Judge
from eval.runner import CaseResult, latency_stats
from eval.scoring import CaseScore, JudgeParseError
from src.llm.client import TrimmedMessage

CASE = GoldenCase(id="c1", category="greeting", input="안녕", expectation="인사한다")

VALID = '{"tone": 4, "worldview": 4, "memory": 4}'
MISSING_AXIS = '{"worldview": 4, "memory": 4}'
GARBAGE = "판정할 수 없습니다"


class _ScriptedClient:
    """정해진 순서대로 응답을 돌려주는 판정자 클라이언트."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def call_llm(self, messages, tools=None, use_stream=False, mute=True, **kwargs):
        self.calls.append(messages[-1]["content"])
        content = self._responses.pop(0) if self._responses else VALID
        return TrimmedMessage(
            content=content, role="assistant", reasoning_content="", tool_calls=[], usage=None
        )


class TestJudgeRetry:
    def test_succeeds_on_first_attempt(self):
        client = _ScriptedClient([VALID])

        score = Judge(client=client).score(CASE, "반갑소")

        assert score.scores["tone"] == 4
        assert len(client.calls) == 1

    def test_retries_when_axis_missing(self):
        """관측된 결함: 판정자가 'tone'을 빠뜨려 사례가 버려졌다."""
        client = _ScriptedClient([MISSING_AXIS, VALID])

        score = Judge(client=client).score(CASE, "반갑소")

        assert score.scores["tone"] == 4
        assert len(client.calls) == 2

    def test_retry_prompt_reminds_format(self):
        client = _ScriptedClient([GARBAGE, VALID])

        Judge(client=client).score(CASE, "반갑소")

        assert "재요청" in client.calls[1]
        assert "재요청" not in client.calls[0]

    def test_gives_up_after_max_attempts(self):
        client = _ScriptedClient([GARBAGE] * (MAX_JUDGE_ATTEMPTS + 2))

        with pytest.raises(JudgeParseError, match=f"{MAX_JUDGE_ATTEMPTS}회 시도"):
            Judge(client=client).score(CASE, "반갑소")

        assert len(client.calls) == MAX_JUDGE_ATTEMPTS

    def test_does_not_retry_valid_response(self):
        """정상 응답에 불필요한 재호출을 하면 판정 비용이 배가된다."""
        client = _ScriptedClient([VALID, VALID, VALID])

        Judge(client=client).score(CASE, "반갑소")

        assert len(client.calls) == 1


class TestLatencyStats:
    def _result(self, case_id: str, latency: float, scored: bool = True) -> CaseResult:
        case = GoldenCase(id=case_id, category="greeting", input="x", expectation="y")
        score = (
            CaseScore(case_id=case_id, category="greeting", scores={"tone": 4}) if scored else None
        )
        return CaseResult(case=case, response="응답", score=score, latency_ms=latency)

    def test_basic_stats(self):
        stats = latency_stats(
            [self._result("a", 1000), self._result("b", 3000), self._result("c", 2000)]
        )

        assert stats["count"] == 3
        assert stats["mean_ms"] == 2000.0
        assert stats["median_ms"] == 2000.0
        assert stats["min_ms"] == 1000.0
        assert stats["max_ms"] == 3000.0

    def test_excludes_failed_cases(self):
        """실패 사례는 재시도로 시간이 왜곡되므로 통계에서 뺀다."""
        stats = latency_stats([self._result("a", 1000), self._result("b", 90000, scored=False)])

        assert stats["count"] == 1
        assert stats["mean_ms"] == 1000.0

    def test_filters_by_ids(self):
        """설정 간 비교는 같은 사례 집합으로만 해야 한다."""
        results = [self._result("a", 1000), self._result("b", 5000)]

        stats = latency_stats(results, only_ids={"a"})

        assert stats["count"] == 1
        assert stats["mean_ms"] == 1000.0

    def test_empty(self):
        assert latency_stats([])["count"] == 0

    def test_even_count_median(self):
        stats = latency_stats([self._result("a", 1000), self._result("b", 2000)])

        assert stats["median_ms"] == 1500.0
