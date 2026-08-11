"""파이프라인 트레이싱 — 각 Stage의 시간, 토큰, 출력 크기를 기록한다."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# StageTrace — 단일 Stage의 실행 기록
# ---------------------------------------------------------------------------


@dataclass
class StageTrace:
    """단일 Stage의 실행 기록."""

    name: str
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    details: dict = field(default_factory=dict)

    def start(self) -> None:
        self.started_at = time.time()

    def finish(self) -> None:
        self.finished_at = time.time()
        self.duration_ms = (self.finished_at - self.started_at) * 1000


# ---------------------------------------------------------------------------
# PipelineTrace — 전체 파이프라인 실행 기록
# ---------------------------------------------------------------------------


@dataclass
class PipelineTrace:
    """전체 파이프라인 실행 기록."""

    user_input: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    total_duration_ms: float = 0.0
    stages: list[StageTrace] = field(default_factory=list)
    response: str = ""
    error: str | None = None
    # LLM 호출 계측 — 호출 수·토큰·라벨별 분해·추정 비용
    metrics: dict = field(default_factory=dict)

    def start(self, user_input: str) -> None:
        self.user_input = user_input
        self.started_at = time.time()

    def finish(self, response: str = "", error: str | None = None) -> None:
        self.finished_at = time.time()
        self.total_duration_ms = (self.finished_at - self.started_at) * 1000
        self.response = response
        self.error = error

    def add_stage(self, name: str) -> StageTrace:
        stage = StageTrace(name=name)
        stage.start()
        self.stages.append(stage)
        return stage

    def to_dict(self) -> dict:
        return {
            "user_input": self.user_input,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "response_length": len(self.response),
            "error": self.error,
            "metrics": self.metrics,
            "stages": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "details": s.details,
                }
                for s in self.stages
            ],
        }


def format_trace(trace: PipelineTrace | None) -> str:
    """트레이스를 사람이 읽는 형태로 출력한다 (`--trace`).

    호출 수·토큰·비용을 라벨별로 보여준다. 어느 단계가 비용을 쓰는지
    한눈에 보이지 않으면 트레이싱의 의미가 없다.
    """
    if trace is None:
        return "(트레이스 없음)"

    lines = [f"── trace ── {trace.total_duration_ms:,.0f}ms"]

    for stage in trace.stages:
        lines.append(f"  [{stage.name}] {stage.duration_ms:,.0f}ms")
        for key, value in stage.details.items():
            lines.append(f"      {key}: {value}")

    m = trace.metrics
    if m:
        from src.pricing import format_cost

        lines.append(f"  [LLM] 호출 {m['calls']}회 · 모델 {m.get('model', '?')}")
        lines.append(
            f"      토큰  입력 {m['prompt_tokens']:,} / 출력 {m['completion_tokens']:,}"
            f" / 합계 {m['total_tokens']:,}"
        )
        lines.append(f"      비용  {format_cost(m.get('cost_usd'))}")
        for label, bucket in m.get("by_label", {}).items():
            lines.append(
                f"      {label:<11} {bucket['calls']}회"
                f"  in {bucket['prompt_tokens']:,} / out {bucket['completion_tokens']:,}"
                f"  {bucket['duration_ms']:,.0f}ms"
            )

    if trace.error:
        lines.append(f"  오류: {trace.error}")

    return "\n".join(lines)
