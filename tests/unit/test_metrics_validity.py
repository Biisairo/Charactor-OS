"""계측의 유효성 — 거부·미상 호출 구별 (TASK-12, REQ-12-1 ~ REQ-12-3, REQ-12-5).

계측은 usage가 없는 호출을 "토큰 0"으로 집계했다. 그런데 토큰 0은 세 가지
서로 다른 상황에서 나온다.

| 상황 | 옳은 해석 |
|---|---|
| 프로바이더가 거부함 | 정확 (과금되지 않음). 다만 거부 사실이 드러나야 한다 |
| 프로바이더가 usage를 생략함 | **틀림** — 실제로는 과금됨 |
| 실제로 토큰을 0개 씀 | 정확 |

셋이 로그에서 같은 모양이면 비용이 과소 추정되었는지 판단할 수 없다.

이 테스트는 LLM API 호출을 하지 않는다.
"""

from __future__ import annotations

import contextlib

from src.metrics import CallMeter, CallRecord

REFUSAL = "The request was rejected because it was considered high risk"


class _Usage:
    def __init__(self, prompt: int, completion: int, total: int):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class _Result:
    def __init__(self, content: str = "안녕하시오", usage: _Usage | None = None):
        self.content = content
        self.usage = usage


class _Client:
    """지정된 결과를 돌려주는 최소 클라이언트."""

    def __init__(self, result: _Result):
        self._result = result

    def call_llm(self, *args, **kwargs):
        return self._result


def _record_one(result: _Result) -> CallRecord:
    meter = CallMeter()
    meter.wrap(_Client(result), "response").call_llm(messages=[], tools=[])
    return meter.records[0]


# ---------------------------------------------------------------------------
# REQ-12-1 — usage 누락은 "토큰 0"이 아니라 미상이다
# ---------------------------------------------------------------------------


class TestUnknownUsage:
    def test_missing_usage_is_marked_unknown(self):
        record = _record_one(_Result(usage=None))
        assert record.usage_known is False

    def test_present_usage_is_marked_known(self):
        record = _record_one(_Result(usage=_Usage(10, 5, 15)))
        assert record.usage_known is True

    def test_genuine_zero_usage_is_known(self):
        """실제로 0을 보고한 것과 usage 자체가 없는 것은 다르다."""
        record = _record_one(_Result(usage=_Usage(0, 0, 0)))
        assert record.usage_known is True
        assert record.prompt_tokens == 0

    def test_failed_call_usage_is_unknown(self):
        class _Failing:
            def call_llm(self, *args, **kwargs):
                raise RuntimeError("boom")

        meter = CallMeter()
        with contextlib.suppress(RuntimeError):
            meter.wrap(_Failing(), "response").call_llm(messages=[], tools=[])

        assert meter.records[0].usage_known is False


# ---------------------------------------------------------------------------
# REQ-12-5 — 거부된 호출이 구별된다
# ---------------------------------------------------------------------------


class TestRefusedCalls:
    def test_refusal_is_marked(self):
        record = _record_one(_Result(content=REFUSAL, usage=_Usage(0, 0, 0)))
        assert record.refused is True

    def test_normal_response_is_not_marked(self):
        record = _record_one(_Result(content="흠, 반갑구나.", usage=_Usage(10, 5, 15)))
        assert record.refused is False

    def test_empty_response_is_marked(self):
        record = _record_one(_Result(content="   ", usage=_Usage(10, 0, 10)))
        assert record.refused is True


# ---------------------------------------------------------------------------
# REQ-12-2 — 턴 집계에 미상·거부 건수가 드러난다
# ---------------------------------------------------------------------------


class TestSummaryExposesValidity:
    def _meter_with(self, results: list[_Result]) -> CallMeter:
        meter = CallMeter()
        for result in results:
            meter.wrap(_Client(result), "response").call_llm(messages=[], tools=[])
        return meter

    def test_summary_counts_unknown_usage(self):
        meter = self._meter_with([_Result(usage=None), _Result(usage=_Usage(10, 5, 15))])
        assert meter.summary()["unknown_usage_calls"] == 1

    def test_summary_counts_refusals(self):
        meter = self._meter_with(
            [_Result(content=REFUSAL, usage=_Usage(0, 0, 0)), _Result(usage=_Usage(10, 5, 15))]
        )
        assert meter.summary()["refused_calls"] == 1

    def test_cost_is_lower_bound_when_usage_unknown(self):
        """미상이 있으면 집계된 토큰은 하한이다. 그 사실이 드러나야 한다."""
        meter = self._meter_with([_Result(usage=None)])
        assert meter.summary()["tokens_are_lower_bound"] is True

    def test_not_lower_bound_when_all_known(self):
        meter = self._meter_with([_Result(usage=_Usage(10, 5, 15))])
        assert meter.summary()["tokens_are_lower_bound"] is False

    def test_clean_turn_reports_zero(self):
        meter = self._meter_with([_Result(usage=_Usage(10, 5, 15))])
        summary = meter.summary()
        assert summary["unknown_usage_calls"] == 0
        assert summary["refused_calls"] == 0

    def test_refused_call_is_not_counted_as_unaccounted(self):
        """거부는 과금되지 않으므로 비용 하한 경고의 근거가 아니다.

        실측에서 확인된 형태다 — 이 프로바이더는 거부 시 usage를 아예 생략한다.
        둘을 뭉뚱그리면 거부가 한 번 날 때마다 "비용 하한" 오경보가 뜬다.
        """
        meter = self._meter_with([_Result(content=REFUSAL, usage=None)])
        summary = meter.summary()

        assert summary["refused_calls"] == 1
        assert summary["unknown_usage_calls"] == 0
        assert summary["tokens_are_lower_bound"] is False

    def test_failed_call_is_not_counted_as_unaccounted(self):
        class _Failing:
            def call_llm(self, *args, **kwargs):
                raise RuntimeError("boom")

        meter = CallMeter()
        with contextlib.suppress(RuntimeError):
            meter.wrap(_Failing(), "response").call_llm(messages=[], tools=[])

        assert meter.summary()["failed_calls"] == 1
        assert meter.summary()["unknown_usage_calls"] == 0

    def test_genuine_omission_is_counted(self):
        """정상 응답인데 usage만 없는 경우 — 이것만이 과소 추정의 원인이다."""
        meter = self._meter_with([_Result(content="흠, 반갑구나.", usage=None)])
        summary = meter.summary()

        assert summary["unknown_usage_calls"] == 1
        assert summary["tokens_are_lower_bound"] is True


# ---------------------------------------------------------------------------
# format_trace 노출
# ---------------------------------------------------------------------------


class TestTraceOutput:
    def test_unknown_usage_is_shown(self):
        from src.trace import PipelineTrace, format_trace

        trace = PipelineTrace()
        trace.start("안녕")
        trace.finish()
        trace.metrics = {
            "calls": 2,
            "failed_calls": 0,
            "refused_calls": 1,
            "unknown_usage_calls": 1,
            "tokens_are_lower_bound": True,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "by_label": {},
            "model": "mock",
        }

        output = format_trace(trace)
        assert "미상" in output
        assert "거부" in output

    def test_clean_turn_does_not_clutter_output(self):
        from src.trace import PipelineTrace, format_trace

        trace = PipelineTrace()
        trace.start("안녕")
        trace.finish()
        trace.metrics = {
            "calls": 1,
            "failed_calls": 0,
            "refused_calls": 0,
            "unknown_usage_calls": 0,
            "tokens_are_lower_bound": False,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "by_label": {},
            "model": "mock",
        }

        output = format_trace(trace)
        assert "미상" not in output
        assert "거부" not in output


class TestToolCallsAreNotRefusals:
    """도구를 부른 응답은 본문이 비어 있다. 그것을 거부로 세면 안 된다 (SPEC-09)."""

    def test_empty_content_with_tool_calls_is_not_refused(self):
        from src.llm.client import ToolCallPart, TrimmedMessage
        from src.metrics import CallMeter

        class _ToolCallingClient:
            def call_llm(self, **kwargs):
                return TrimmedMessage(
                    content="",
                    role="assistant",
                    reasoning_content="",
                    tool_calls=[ToolCallPart(id="1", name="search_memory", arguments="{}")],
                    usage=None,
                )

        meter = CallMeter()
        meter.wrap(_ToolCallingClient(), "react").call_llm(messages=[], tools=[])

        assert meter.summary()["refused_calls"] == 0

    def test_empty_content_without_tool_calls_is_still_refused(self):
        from src.llm.client import TrimmedMessage
        from src.metrics import CallMeter

        class _EmptyClient:
            def call_llm(self, **kwargs):
                return TrimmedMessage(
                    content="", role="assistant", reasoning_content="", tool_calls=[], usage=None
                )

        meter = CallMeter()
        meter.wrap(_EmptyClient(), "response").call_llm(messages=[], tools=[])

        assert meter.summary()["refused_calls"] == 1
