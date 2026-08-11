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
            "stages": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "details": s.details,
                }
                for s in self.stages
            ],
        }
