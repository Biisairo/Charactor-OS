"""파이프라인 계측 통합 검증 (TASK-04 / REQ-04-1 ~ 04-5).

계측은 트레이싱을 켤 때만 동작하고, 켜든 끄든 대화 결과는 같아야 한다.
켰을 때는 턴당 호출 수·토큰·비용이 트레이스로 조회 가능해야 한다.
"""

from __future__ import annotations

from pathlib import Path

from src.llm.client import TokenUsage, TrimmedMessage
from src.trace import format_trace
from tests.conftest import PipelineMockClient, make_character_os


class MeteredMockClient(PipelineMockClient):
    """토큰 사용량을 함께 돌려주는 MockClient."""

    def __init__(self, model: str = "gpt-4o-mini", prompt: int = 1000, completion: int = 50):
        super().__init__()
        self.env = type("Env", (), {"model": model})()
        self._prompt = prompt
        self._completion = completion

    def call_llm(self, messages, tools=None, use_stream=False, mute=True, **kwargs):
        result = super().call_llm(messages, tools=tools, use_stream=use_stream, mute=mute, **kwargs)
        return TrimmedMessage(
            content=result.content,
            role=result.role,
            reasoning_content=result.reasoning_content,
            tool_calls=result.tool_calls,
            usage=TokenUsage(
                prompt_tokens=self._prompt,
                completion_tokens=self._completion,
                total_tokens=self._prompt + self._completion,
            ),
        )


class TestTraceDisabledByDefault:
    def test_no_trace_snapshot_without_flag(self, character_dir: Path, tmp_path: Path):
        """`--trace` 없이는 트레이스를 만들지 않는다."""
        cos = make_character_os(character_dir, tmp_path, MeteredMockClient())

        cos.chat("안녕하시오")

        assert cos._last_trace is None

    def test_metrics_still_collected_for_ops_log(self, character_dir: Path, tmp_path: Path):
        """디버그 플래그와 무관하게 계측은 돌아야 한다 — 운영 로그가 이에 의존한다."""
        cos = make_character_os(character_dir, tmp_path, MeteredMockClient())

        cos.chat("안녕하시오")

        assert cos._meter.summary()["calls"] == 3

    def test_conversation_result_is_identical(self, character_dir: Path, tmp_path: Path):
        """계측 여부가 대화 동작을 바꾸면 안 된다 (REQ-04-5)."""
        off = make_character_os(character_dir, tmp_path / "a", MeteredMockClient())
        on = make_character_os(character_dir, tmp_path / "b", MeteredMockClient(), trace=True)

        assert off.chat("안녕하시오") == on.chat("안녕하시오")


class TestMeteringEnabled:
    def _run(self, character_dir: Path, tmp_path: Path, **kwargs):
        cos = make_character_os(character_dir, tmp_path, MeteredMockClient(), trace=True, **kwargs)
        cos.chat("안녕하시오")
        return cos._last_trace.metrics

    def test_counts_calls(self, character_dir: Path, tmp_path: Path):
        """초안 1 + 감정 1 + 기억 1 = 3회 (Reflection 비활성)."""
        assert self._run(character_dir, tmp_path)["calls"] == 3

    def test_sums_tokens(self, character_dir: Path, tmp_path: Path):
        metrics = self._run(character_dir, tmp_path)

        assert metrics["prompt_tokens"] == 3000
        assert metrics["completion_tokens"] == 150
        assert metrics["total_tokens"] == 3150

    def test_separates_call_sites(self, character_dir: Path, tmp_path: Path):
        """어느 단계가 호출했는지 구분되어야 한다 (REQ-04-1)."""
        by_label = self._run(character_dir, tmp_path)["by_label"]

        assert set(by_label) == {"response", "emotion", "memory"}

    def test_reflection_adds_calls(self, character_dir: Path, tmp_path: Path):
        """Reflection을 켜면 검토 호출이 추가로 잡혀야 한다."""
        without = self._run(character_dir, tmp_path / "a")
        with_review = self._run(character_dir, tmp_path / "b", no_review=False)

        assert with_review["calls"] > without["calls"]
        assert "reflection" in with_review["by_label"]

    def test_estimates_cost(self, character_dir: Path, tmp_path: Path):
        """등록된 모델은 비용이 산출되어야 한다 (REQ-04-3)."""
        metrics = self._run(character_dir, tmp_path)

        assert metrics["model"] == "gpt-4o-mini"
        assert metrics["cost_usd"] is not None
        assert metrics["cost_usd"] > 0

    def test_unknown_model_has_no_cost(self, character_dir: Path, tmp_path: Path):
        """단가 미등록 모델을 0원으로 보고하면 안 된다."""
        cos = make_character_os(
            character_dir, tmp_path, MeteredMockClient(model="등록안된모델"), trace=True
        )
        cos.chat("안녕하시오")

        assert cos._last_trace.metrics["cost_usd"] is None

    def test_metrics_reset_each_turn(self, character_dir: Path, tmp_path: Path):
        """턴 단위 집계여야 한다. 누적되면 턴당 비용을 알 수 없다."""
        cos = make_character_os(character_dir, tmp_path, MeteredMockClient(), trace=True)

        cos.chat("안녕하시오")
        first = cos._last_trace.metrics["calls"]
        cos.chat("또 뵙는구려")
        second = cos._last_trace.metrics["calls"]

        assert first == second

    def test_metrics_present_in_serialized_trace(self, character_dir: Path, tmp_path: Path):
        """`GET /api/trace/last`가 이 dict를 그대로 반환한다 (REQ-04-4)."""
        cos = make_character_os(character_dir, tmp_path, MeteredMockClient(), trace=True)
        cos.chat("안녕하시오")

        payload = cos._last_trace.to_dict()

        assert payload["metrics"]["calls"] == 3
        assert "cost_usd" in payload["metrics"]


class TestFormatTrace:
    def test_includes_calls_tokens_and_cost(self, character_dir: Path, tmp_path: Path):
        """`--trace` 출력에 호출 수·토큰·비용이 보여야 한다 (인수 기준 1)."""
        cos = make_character_os(character_dir, tmp_path, MeteredMockClient(), trace=True)
        cos.chat("안녕하시오")

        text = format_trace(cos._last_trace)

        assert "호출 3회" in text
        assert "토큰" in text
        assert "비용" in text
        assert "response" in text

    def test_unknown_model_shows_placeholder(self, character_dir: Path, tmp_path: Path):
        cos = make_character_os(
            character_dir, tmp_path, MeteredMockClient(model="미등록"), trace=True
        )
        cos.chat("안녕하시오")

        assert "단가 미등록" in format_trace(cos._last_trace)

    def test_handles_none(self):
        assert format_trace(None) == "(트레이스 없음)"
