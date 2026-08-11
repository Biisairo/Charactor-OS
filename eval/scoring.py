"""채점 결과의 파싱·집계·비교 (REQ-01-3, 01-4, 01-6, 01-8).

이 모듈에는 LLM 호출이 없다. 판정 결과를 다루는 로직만 담아
API 키 없이 단위 테스트할 수 있게 분리했다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# 채점 축 — 키, 표시 이름, 판정 기준
AXES: dict[str, str] = {
    "tone": "말투 일관성",
    "worldview": "세계관 정합성",
    "memory": "기억 활용 적절성",
}

MIN_SCORE = 1
MAX_SCORE = 5


class JudgeParseError(ValueError):
    """판정자 응답을 해석할 수 없을 때 발생한다."""


@dataclass
class CaseScore:
    """단일 사례의 채점 결과.

    scores는 판정 시 정수(1~5)지만, 여러 실행을 평균하면 실수가 된다.
    """

    case_id: str
    category: str
    scores: dict[str, float]
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return sum(self.scores.values()) / len(self.scores)


def average_across_runs(runs: list[list[CaseScore]]) -> list[CaseScore]:
    """여러 실행의 사례별 점수를 평균한다.

    1회 실행의 점수는 판정자 변동 때문에 흔들린다. 같은 설정을 N회 반복해
    사례별로 평균하면 그 변동이 줄어, 설정 간 차이를 노이즈와 구분할 수 있다.

    **모든 실행에서 채점된 사례만** 남긴다. 일부 실행에서만 성공한 사례를 섞으면
    실행마다 표본이 달라져 평균의 의미가 흐려진다.
    """
    if not runs:
        return []

    by_id: dict[str, list[CaseScore]] = {}
    for run in runs:
        for cs in run:
            by_id.setdefault(cs.case_id, []).append(cs)

    averaged: list[CaseScore] = []
    for case_id, scores in sorted(by_id.items()):
        if len(scores) != len(runs):
            continue
        averaged.append(
            CaseScore(
                case_id=case_id,
                category=scores[0].category,
                scores={
                    axis: round(sum(s.scores[axis] for s in scores) / len(scores), 3)
                    for axis in AXES
                },
            )
        )
    return averaged


@dataclass
class Summary:
    """축별 평균과 전체 평균."""

    per_axis: dict[str, float]
    overall: float
    case_count: int
    per_category: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "per_axis": self.per_axis,
            "overall": self.overall,
            "case_count": self.case_count,
            "per_category": self.per_category,
        }


@dataclass
class AxisDelta:
    """두 설정 간 축별 점수 차이."""

    axis: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return round(self.after - self.before, 3)


def _round(value: float) -> float:
    return round(value, 3)


def parse_judge_response(raw: str) -> tuple[dict[str, int], dict[str, str]]:
    """판정자의 JSON 응답에서 점수와 근거를 추출한다.

    모델이 JSON 앞뒤에 설명을 붙이거나 코드 펜스로 감싸는 경우가 흔하므로,
    첫 번째 JSON 객체를 찾아 파싱한다.

    Raises:
        JudgeParseError: JSON을 찾을 수 없거나, 축이 빠졌거나, 점수가 범위를 벗어난 경우.
    """
    payload = _extract_json_object(raw)

    scores: dict[str, int] = {}
    reasons: dict[str, str] = {}

    for axis in AXES:
        if axis not in payload:
            raise JudgeParseError(f"판정 결과에 '{axis}' 축이 없습니다: {raw[:200]}")

        entry = payload[axis]
        # {"tone": 4} 와 {"tone": {"score": 4, "reason": "..."}} 를 모두 받는다
        if isinstance(entry, dict):
            raw_score = entry.get("score")
            reasons[axis] = str(entry.get("reason", "")).strip()
        else:
            raw_score = entry

        score = _coerce_score(axis, raw_score)
        scores[axis] = score

    return scores, reasons


def _extract_json_object(raw: str) -> dict:
    text = raw.strip()
    # 코드 펜스 제거
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise JudgeParseError(f"판정 결과에서 JSON을 찾을 수 없습니다: {raw[:200]}") from None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise JudgeParseError(f"판정 결과 JSON 파싱 실패: {exc}") from exc

    if not isinstance(payload, dict):
        raise JudgeParseError(f"판정 결과가 객체가 아닙니다: {raw[:200]}")
    return payload


def _coerce_score(axis: str, value: object) -> int:
    if isinstance(value, bool) or value is None:
        raise JudgeParseError(f"'{axis}' 점수가 숫자가 아닙니다: {value!r}")
    if isinstance(value, str):
        value = value.strip()
    try:
        score = int(round(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise JudgeParseError(f"'{axis}' 점수가 숫자가 아닙니다: {value!r}") from None

    if not MIN_SCORE <= score <= MAX_SCORE:
        raise JudgeParseError(
            f"'{axis}' 점수가 범위를 벗어났습니다: {score} (허용 {MIN_SCORE}~{MAX_SCORE})"
        )
    return score


def aggregate(case_scores: list[CaseScore]) -> Summary:
    """축별 평균, 전체 평균, 범주별 평균을 계산한다."""
    if not case_scores:
        return Summary(per_axis=dict.fromkeys(AXES, 0.0), overall=0.0, case_count=0)

    per_axis = {
        axis: _round(sum(cs.scores[axis] for cs in case_scores) / len(case_scores)) for axis in AXES
    }
    overall = _round(sum(per_axis.values()) / len(per_axis))

    by_category: dict[str, list[float]] = {}
    for cs in case_scores:
        by_category.setdefault(cs.category, []).append(cs.mean)
    per_category = {cat: _round(sum(vals) / len(vals)) for cat, vals in sorted(by_category.items())}

    return Summary(
        per_axis=per_axis,
        overall=overall,
        case_count=len(case_scores),
        per_category=per_category,
    )


@dataclass
class PairedComparison:
    """공통 사례만으로 두 설정을 비교한 결과."""

    before: Summary
    after: Summary
    common_ids: list[str]
    dropped_before: list[str]
    dropped_after: list[str]
    wins_after: int
    wins_before: int
    ties: int

    @property
    def deltas(self) -> list[AxisDelta]:
        return compare(self.before, self.after)

    def to_dict(self) -> dict:
        return {
            "common_case_count": len(self.common_ids),
            "common_ids": self.common_ids,
            "dropped_from_before": self.dropped_before,
            "dropped_from_after": self.dropped_after,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "deltas": {d.axis: d.delta for d in self.deltas},
            "case_wins": {
                "after": self.wins_after,
                "before": self.wins_before,
                "tie": self.ties,
            },
        }


def paired_compare(before: list[CaseScore], after: list[CaseScore]) -> PairedComparison:
    """두 설정을 **공통으로 채점된 사례만으로** 비교한다.

    설정마다 채점 실패 건수가 다르면 사례 집합이 달라진다. 그 상태로 전체 평균을
    비교하면 설정의 효과가 아니라 표본 차이를 보게 된다. 실제로 한 설정에서만
    프로바이더 거부가 4건 발생해 비교가 뒤집힌 적이 있어, 교집합으로 강제한다.
    """
    by_id_before = {cs.case_id: cs for cs in before}
    by_id_after = {cs.case_id: cs for cs in after}
    common = sorted(set(by_id_before) & set(by_id_after))

    wins_after = wins_before = ties = 0
    for case_id in common:
        b = sum(by_id_before[case_id].scores.values())
        a = sum(by_id_after[case_id].scores.values())
        if a > b:
            wins_after += 1
        elif a < b:
            wins_before += 1
        else:
            ties += 1

    return PairedComparison(
        before=aggregate([by_id_before[c] for c in common]),
        after=aggregate([by_id_after[c] for c in common]),
        common_ids=common,
        dropped_before=sorted(set(by_id_before) - set(common)),
        dropped_after=sorted(set(by_id_after) - set(common)),
        wins_after=wins_after,
        wins_before=wins_before,
        ties=ties,
    )


def compare(before: Summary, after: Summary) -> list[AxisDelta]:
    """두 설정의 축별 차이를 계산한다. 전체 평균은 'overall' 항목으로 포함한다."""
    deltas = [
        AxisDelta(
            axis=axis, before=before.per_axis.get(axis, 0.0), after=after.per_axis.get(axis, 0.0)
        )
        for axis in AXES
    ]
    deltas.append(AxisDelta(axis="overall", before=before.overall, after=after.overall))
    return deltas


def run_spread(summaries: list[Summary]) -> dict:
    """같은 설정을 반복 실행했을 때의 전체 평균 변동 폭.

    이 값이 설정 간 차이보다 크면, 그 차이는 노이즈와 구분할 수 없다.
    """
    values = [s.overall for s in summaries]
    if len(values) < 2:
        return {"runs": len(values), "spread": 0.0, "values": values}
    return {
        "runs": len(values),
        "min": _round(min(values)),
        "max": _round(max(values)),
        "spread": _round(max(values) - min(values)),
        "mean": _round(sum(values) / len(values)),
        "values": [_round(v) for v in values],
    }


def format_summary(summary: Summary) -> str:
    """사람이 읽는 요약 표."""
    lines = [f"사례 {summary.case_count}건"]
    for axis, label in AXES.items():
        lines.append(f"  {label:<12} {summary.per_axis[axis]:.2f}")
    lines.append(f"  {'전체':<12} {summary.overall:.2f}")
    if summary.per_category:
        lines.append("  범주별:")
        for cat, value in summary.per_category.items():
            lines.append(f"    {cat:<20} {value:.2f}")
    return "\n".join(lines)


def format_comparison(label_before: str, label_after: str, deltas: list[AxisDelta]) -> str:
    """두 설정 비교 표."""
    lines = [f"{'축':<14}{label_before:>10}{label_after:>10}{'차이':>10}"]
    lines.append("-" * 44)
    for d in deltas:
        name = AXES.get(d.axis, d.axis if d.axis != "overall" else "전체")
        sign = "+" if d.delta > 0 else ""
        lines.append(f"{name:<14}{d.before:>10.2f}{d.after:>10.2f}{sign + f'{d.delta:.2f}':>10}")
    return "\n".join(lines)
