"""LLM 호출 계측 (TASK-04 / REQ-04-1, 04-2, 04-5).

호출 지점마다 계측 코드를 심으면 한 곳만 빠뜨려도 집계가 조용히 틀린다.
클라이언트를 감싸는 방식이 실제로 모든 호출을 포착하는지, 그리고
계측이 대화 동작을 바꾸지 않는지 검증한다.
"""

from __future__ import annotations

from src.llm.client import TokenUsage, TrimmedMessage
from src.metrics import CallMeter, CallRecord


class _Client:
    """호출을 세는 더미 클라이언트."""

    env = type("Env", (), {"model": "test-model"})()

    def __init__(self, usage: TokenUsage | None = None):
        self.calls = 0
        self._usage = usage
        self.extra_attribute = "원본 속성"

    def call_llm(self, messages, tools=None, use_stream=False, mute=True, **kwargs):
        self.calls += 1
        return TrimmedMessage(
            content="응답",
            role="assistant",
            reasoning_content="",
            tool_calls=[],
            usage=self._usage,
        )


def _usage(p: int = 100, c: int = 20) -> TokenUsage:
    return TokenUsage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


class TestSinkIsOptional:
    def test_works_without_sink(self):
        """싱크가 없어도 턴 스냅샷은 모아야 한다."""
        meter = CallMeter()
        meter.record(CallRecord("response", 1, 1, 2, 1.0))

        assert meter.summary()["calls"] == 1

    def test_sink_receives_every_call(self):
        seen = []
        meter = CallMeter(sink=seen.append)
        meter.wrap(_Client(_usage()), "response").call_llm(messages=[])

        assert len(seen) == 1

    def test_sink_failure_does_not_break_conversation(self):
        """로그가 깨져도 대화는 계속되어야 한다."""

        def broken(_record):
            raise OSError("디스크 오류")

        meter = CallMeter(sink=broken)
        result = meter.wrap(_Client(_usage()), "response").call_llm(messages=[])

        assert result.content == "응답"
        assert meter.summary()["calls"] == 1


class TestTransparency:
    def test_response_is_unchanged(self):
        """계측이 응답 내용을 바꾸면 안 된다."""
        client = _Client(_usage())
        wrapped = CallMeter().wrap(client, "response")

        result = wrapped.call_llm(messages=[{"role": "user", "content": "안녕"}])

        assert result.content == "응답"

    def test_underlying_client_is_called_once(self):
        client = _Client(_usage())
        wrapped = CallMeter().wrap(client, "response")

        wrapped.call_llm(messages=[])

        assert client.calls == 1

    def test_other_attributes_delegate(self):
        """감싼 뒤에도 원본의 다른 속성에 접근할 수 있어야 한다."""
        client = _Client()
        wrapped = CallMeter().wrap(client, "response")

        assert wrapped.extra_attribute == "원본 속성"
        assert wrapped.env.model == "test-model"


class TestAggregation:
    def test_counts_calls(self):
        meter = CallMeter()
        wrapped = meter.wrap(_Client(_usage()), "response")

        for _ in range(3):
            wrapped.call_llm(messages=[])

        assert meter.summary()["calls"] == 3

    def test_sums_tokens(self):
        meter = CallMeter()
        wrapped = meter.wrap(_Client(_usage(p=100, c=20)), "response")

        wrapped.call_llm(messages=[])
        wrapped.call_llm(messages=[])

        summary = meter.summary()
        assert summary["prompt_tokens"] == 200
        assert summary["completion_tokens"] == 40
        assert summary["total_tokens"] == 240

    def test_separates_by_label(self):
        """어느 단계가 비용을 쓰는지 구분되어야 한다."""
        meter = CallMeter()
        client = _Client(_usage())

        meter.wrap(client, "response").call_llm(messages=[])
        meter.wrap(client, "emotion").call_llm(messages=[])
        meter.wrap(client, "emotion").call_llm(messages=[])

        by_label = meter.summary()["by_label"]
        assert by_label["response"]["calls"] == 1
        assert by_label["emotion"]["calls"] == 2

    def test_missing_usage_counts_as_zero(self):
        """프로바이더가 usage를 주지 않아도 호출 수는 세어야 한다."""
        meter = CallMeter()
        meter.wrap(_Client(usage=None), "response").call_llm(messages=[])

        summary = meter.summary()
        assert summary["calls"] == 1
        assert summary["total_tokens"] == 0

    def test_records_duration(self):
        meter = CallMeter()
        meter.wrap(_Client(_usage()), "response").call_llm(messages=[])

        assert meter.records[0].duration_ms >= 0

    def test_empty_summary(self):
        summary = CallMeter().summary()

        assert summary["calls"] == 0
        assert summary["by_label"] == {}


class TestReset:
    def test_reset_clears_records(self):
        """계측은 턴 단위다. 초기화하지 않으면 이전 턴이 섞인다."""
        meter = CallMeter()
        wrapped = meter.wrap(_Client(_usage()), "response")
        wrapped.call_llm(messages=[])

        meter.reset()

        assert meter.summary()["calls"] == 0

    def test_wrapped_client_still_works_after_reset(self):
        meter = CallMeter()
        wrapped = meter.wrap(_Client(_usage()), "response")
        wrapped.call_llm(messages=[])
        meter.reset()
        wrapped.call_llm(messages=[])

        assert meter.summary()["calls"] == 1
