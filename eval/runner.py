"""평가 실행 — 캐릭터 응답 생성 → 판정 → 집계 → 저장 (REQ-01-4, 01-5, 01-11)."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from eval.dataset import GoldenCase, GoldenDataset
from eval.scoring import CaseScore, Summary, aggregate
from src.character_os import CharacterOS
from src.config import load_config
from src.prompts.engine import from_config as prompt_engine_from_config
from src.validity import provider_error_reason

RESULTS_DIR = Path(__file__).parent / "results"

# 프로바이더 거부는 일시적일 수 있다. 재시도로 표본 손실과 설정 간 편향을 줄인다.
MAX_GENERATION_ATTEMPTS = 3


def current_commit() -> str:
    """평가 대상 커밋. 결과 재현을 위해 기록한다 (REQ-01-5)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(unknown)"


@dataclass
class CaseResult:
    case: GoldenCase
    response: str
    score: CaseScore | None
    error: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.case.id,
            "category": self.case.category,
            "input": self.case.input,
            "setup": self.case.setup,
            "response": self.response,
            "latency_ms": round(self.latency_ms, 1),
            "scores": self.score.scores if self.score else None,
            "reasons": self.score.reasons if self.score else None,
            "error": self.error,
        }


def latency_stats(results: list[CaseResult], only_ids: set[str] | None = None) -> dict:
    """채점에 성공한 사례들의 응답 지연 통계.

    실패 사례는 재시도로 시간이 왜곡되므로 제외한다.
    only_ids를 주면 그 사례들로만 계산한다 (설정 간 짝 비교용).
    """
    values = sorted(
        r.latency_ms
        for r in results
        if r.score is not None and (only_ids is None or r.case.id in only_ids)
    )
    if not values:
        return {"count": 0}

    n = len(values)
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    return {
        "count": n,
        "mean_ms": round(sum(values) / n, 1),
        "median_ms": round(median, 1),
        "min_ms": round(values[0], 1),
        "max_ms": round(values[-1], 1),
    }


@dataclass
class EvalRun:
    """단일 설정에 대한 평가 실행 결과."""

    character: str
    setting: str
    target_model: str
    judge_model: str
    commit: str
    timestamp: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def scored(self) -> list[CaseScore]:
        return [r.score for r in self.results if r.score is not None]

    @property
    def summary(self) -> Summary:
        return aggregate(self.scored)

    def to_dict(self) -> dict:
        return {
            "character": self.character,
            "setting": self.setting,
            "target_model": self.target_model,
            "judge_model": self.judge_model,
            "commit": self.commit,
            "timestamp": self.timestamp,
            "summary": self.summary.to_dict(),
            "latency": latency_stats(self.results),
            "excluded_count": len(self.results) - len(self.scored),
            "failed_cases": [
                {"id": r.case.id, "reason": r.error} for r in self.results if r.score is None
            ],
            "cases": [r.to_dict() for r in self.results],
        }

    def filename_stem(self) -> str:
        """실행마다 고유한 파일명.

        고정 이름을 쓰면 재실행이 이전 결과를 조용히 덮어쓴다. 실제로 그렇게
        해서 어느 실행의 수치인지 알 수 없게 된 적이 있어, 실행 시각을 이름에
        포함한다. 이력이 남고 동시 실행도 충돌하지 않는다.
        """
        stamp = self.timestamp.replace(":", "").replace("-", "").replace("T", "-")
        return f"{self.character}_{self.setting}_{stamp}"

    def save(self, out_dir: Path | None = None) -> Path:
        """결과를 저장하고, 같은 설정의 최신 결과를 가리키는 사본도 갱신한다."""
        directory = out_dir or RESULTS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

        path = directory / f"{self.filename_stem()}.json"
        path.write_text(payload, encoding="utf-8")

        # README가 참조할 안정적인 경로 — 항상 가장 최근 실행을 가리킨다
        latest = directory / f"{self.character}_{self.setting}_latest.json"
        latest.write_text(payload, encoding="utf-8")

        return path


def _generate_response(
    case: GoldenCase,
    character_dir: Path,
    state_dir: Path,
    no_review: bool,
    client,
    attempt: int,
) -> tuple[str, str, float]:
    """사례 1회 시도. (응답, 오류사유, 지연ms)를 반환하며 오류사유가 비면 성공이다.

    지연은 **평가 대상 턴 하나**만 측정한다. setup 발화는 기억을 심기 위한
    준비 과정이므로 응답 지연에 포함하지 않는다.
    """
    kwargs = {"client": client} if client is not None else {}
    workdir = state_dir / f"attempt-{attempt}"
    workdir.mkdir(parents=True, exist_ok=True)

    cos = CharacterOS(
        character_dir=str(character_dir),
        memory_db_path=str(workdir / "memories.db"),
        emotion_save_path=str(workdir / "emotions.json"),
        history_save_path=str(workdir / "history.json"),
        working_memory_path=str(workdir / "working_memory.json"),
        debug=False,
        output=lambda _msg: None,
        no_review=no_review,
        # 평가만 다른 자로 재면 측정이 런타임과 어긋난다 (SPEC-11 REQ-11-11).
        prompt_engine=prompt_engine_from_config(load_config()),
        **kwargs,
    )

    try:
        for utterance in case.setup:
            cos.chat(utterance)

        started = time.perf_counter()
        response = cos.chat(case.input)
        latency_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:  # noqa: BLE001 — 사례 하나의 실패가 전체를 막지 않는다
        return "", f"대화 실패: {exc}", 0.0

    if not response:
        return "", "응답 생성 실패", latency_ms

    # 프로바이더 거부 응답을 채점하면 인프라 문제가 품질 점수로 둔갑한다
    invalid = provider_error_reason(response)
    if invalid:
        return response, invalid, latency_ms

    return response, "", latency_ms


def run_case(
    case: GoldenCase,
    character_dir: Path,
    state_dir: Path,
    judge,
    no_review: bool,
    client=None,
) -> CaseResult:
    """사례 1건을 평가한다. setup 발화를 먼저 대화한 뒤 평가 입력을 넣는다.

    프로바이더 거부는 일시적이므로 재시도한다. 재시도 없이 버리면 표본이 줄고,
    거부가 특정 설정에 몰리면 설정 간 비교가 편향된다.
    """
    response, error, latency_ms = "", "", 0.0
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        response, error, latency_ms = _generate_response(
            case, character_dir, state_dir, no_review, client, attempt
        )
        if not error:
            break

    if error:
        suffix = f" ({MAX_GENERATION_ATTEMPTS}회 시도)" if MAX_GENERATION_ATTEMPTS > 1 else ""
        return CaseResult(
            case=case, response=response, score=None, error=error + suffix, latency_ms=latency_ms
        )

    try:
        score = judge.score(case, response)
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            case=case,
            response=response,
            score=None,
            error=f"판정 실패: {exc}",
            latency_ms=latency_ms,
        )

    return CaseResult(case=case, response=response, score=score, latency_ms=latency_ms)


def run_evaluation(
    dataset: GoldenDataset,
    character_dir: Path,
    workspace: Path,
    judge,
    *,
    setting: str,
    no_review: bool,
    target_model: str,
    judge_model: str,
    client=None,
    limit: int | None = None,
    progress=None,
) -> EvalRun:
    """데이터셋 전체를 평가한다."""
    cases = dataset.cases[:limit] if limit else dataset.cases

    run = EvalRun(
        character=dataset.character,
        setting=setting,
        target_model=target_model,
        judge_model=judge_model,
        commit=current_commit(),
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )

    for index, case in enumerate(cases, start=1):
        state_dir = workspace / setting / case.id
        state_dir.mkdir(parents=True, exist_ok=True)

        result = run_case(case, character_dir, state_dir, judge, no_review, client=client)
        run.results.append(result)

        if progress:
            progress(index, len(cases), result)

    return run
