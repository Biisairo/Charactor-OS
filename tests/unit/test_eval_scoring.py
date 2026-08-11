"""평가 채점 로직 단위 테스트 (TASK-01 / REQ-01-8).

파싱·집계·비교는 LLM 호출 없이 검증할 수 있어야 한다. 판정자 응답은
모델이 만드는 자유 텍스트이므로, 형식이 흔들려도 견디는지가 핵심이다.
"""

from __future__ import annotations

import pytest

from eval.scoring import (
    AXES,
    AxisDelta,
    CaseScore,
    JudgeParseError,
    Summary,
    aggregate,
    compare,
    format_comparison,
    format_summary,
    parse_judge_response,
)


def _score(case_id: str, category: str, tone: int, worldview: int, memory: int) -> CaseScore:
    return CaseScore(
        case_id=case_id,
        category=category,
        scores={"tone": tone, "worldview": worldview, "memory": memory},
    )


# ---------------------------------------------------------------------------
# 판정 응답 파싱 — 형식 변형에 견뎌야 한다
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    def test_nested_form(self):
        raw = """{
          "tone": {"score": 5, "reason": "말투 일관"},
          "worldview": {"score": 4, "reason": "설정 부합"},
          "memory": {"score": 3, "reason": "부분 활용"}
        }"""
        scores, reasons = parse_judge_response(raw)

        assert scores == {"tone": 5, "worldview": 4, "memory": 3}
        assert reasons["tone"] == "말투 일관"

    def test_flat_form(self):
        """점수만 반환하는 축약형도 받는다."""
        scores, reasons = parse_judge_response('{"tone": 4, "worldview": 5, "memory": 2}')

        assert scores == {"tone": 4, "worldview": 5, "memory": 2}
        assert reasons == {}

    def test_code_fence_is_stripped(self):
        raw = '```json\n{"tone": 3, "worldview": 3, "memory": 3}\n```'
        scores, _ = parse_judge_response(raw)

        assert scores["tone"] == 3

    def test_surrounding_prose_is_tolerated(self):
        raw = '평가 결과입니다.\n{"tone": 2, "worldview": 2, "memory": 2}\n이상입니다.'
        scores, _ = parse_judge_response(raw)

        assert scores["worldview"] == 2

    def test_numeric_string_is_coerced(self):
        scores, _ = parse_judge_response('{"tone": "4", "worldview": 3, "memory": 3}')

        assert scores["tone"] == 4

    def test_float_is_rounded(self):
        scores, _ = parse_judge_response('{"tone": 4.4, "worldview": 3, "memory": 3}')

        assert scores["tone"] == 4

    @pytest.mark.parametrize(
        "raw",
        [
            '{"tone": 4, "worldview": 3}',  # memory 누락
            '{"worldview": 3, "memory": 3}',  # tone 누락
        ],
    )
    def test_missing_axis_rejected(self, raw: str):
        with pytest.raises(JudgeParseError, match="축이 없습니다"):
            parse_judge_response(raw)

    @pytest.mark.parametrize("value", [0, 6, -1, 99])
    def test_out_of_range_rejected(self, value: int):
        raw = f'{{"tone": {value}, "worldview": 3, "memory": 3}}'
        with pytest.raises(JudgeParseError, match="범위를 벗어났"):
            parse_judge_response(raw)

    @pytest.mark.parametrize(
        "raw",
        [
            "JSON이 전혀 없는 응답",
            '{"tone": "훌륭함", "worldview": 3, "memory": 3}',
            '{"tone": null, "worldview": 3, "memory": 3}',
            "[1, 2, 3]",
        ],
    )
    def test_malformed_rejected(self, raw: str):
        with pytest.raises(JudgeParseError):
            parse_judge_response(raw)


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_axis_means(self):
        summary = aggregate(
            [
                _score("a", "greeting", 5, 4, 3),
                _score("b", "greeting", 3, 2, 1),
            ]
        )

        assert summary.per_axis == {"tone": 4.0, "worldview": 3.0, "memory": 2.0}
        assert summary.case_count == 2

    def test_overall_is_mean_of_axes(self):
        summary = aggregate([_score("a", "greeting", 5, 4, 3)])

        assert summary.overall == 4.0

    def test_per_category(self):
        summary = aggregate(
            [
                _score("a", "greeting", 5, 5, 5),
                _score("b", "greeting", 3, 3, 3),
                _score("c", "worldview", 1, 1, 1),
            ]
        )

        assert summary.per_category == {"greeting": 4.0, "worldview": 1.0}

    def test_empty_input(self):
        summary = aggregate([])

        assert summary.case_count == 0
        assert summary.overall == 0.0
        assert set(summary.per_axis) == set(AXES)

    def test_rounding_is_stable(self):
        """3으로 나누어떨어지지 않는 값도 결정론적으로 반올림된다."""
        summary = aggregate(
            [
                _score("a", "x", 5, 5, 5),
                _score("b", "x", 4, 4, 4),
                _score("c", "x", 4, 4, 4),
            ]
        )

        assert summary.per_axis["tone"] == 4.333


# ---------------------------------------------------------------------------
# 비교
# ---------------------------------------------------------------------------


class TestCompare:
    def test_delta_direction(self):
        before = aggregate([_score("a", "x", 3, 3, 3)])
        after = aggregate([_score("a", "x", 4, 4, 4)])

        deltas = {d.axis: d.delta for d in compare(before, after)}

        assert deltas["tone"] == 1.0
        assert deltas["overall"] == 1.0

    def test_negative_delta(self):
        before = aggregate([_score("a", "x", 5, 5, 5)])
        after = aggregate([_score("a", "x", 2, 2, 2)])

        assert compare(before, after)[0].delta == -3.0

    def test_includes_overall_row(self):
        deltas = compare(aggregate([]), aggregate([]))

        assert [d.axis for d in deltas] == [*AXES.keys(), "overall"]


# ---------------------------------------------------------------------------
# 출력 형식 — 사람이 읽는 표가 깨지지 않는지만 확인한다
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_summary_contains_axis_labels(self):
        text = format_summary(aggregate([_score("a", "greeting", 5, 4, 3)]))

        for label in AXES.values():
            assert label in text

    def test_comparison_marks_positive_delta(self):
        text = format_comparison("off", "on", [AxisDelta(axis="tone", before=3.0, after=4.0)])

        assert "+1.00" in text

    def test_summary_to_dict_roundtrip(self):
        summary = Summary(per_axis={"tone": 4.0}, overall=4.0, case_count=1)

        assert summary.to_dict()["overall"] == 4.0
