"""설정 간 짝 비교 검증 (TASK-01 후속).

설정마다 채점 실패 수가 다르면 사례 집합이 어긋난다. 그 상태로 전체 평균을
비교하면 설정의 효과가 아니라 표본 차이를 보게 된다.

실제로 한 설정에서만 프로바이더 거부가 4건 발생해 비교 부호가 뒤집힌 적이 있다.
(off 4.64 vs on 4.30 → 짝 비교하면 off 4.64 vs on 4.36)
"""

from __future__ import annotations

from eval.scoring import CaseScore, paired_compare


def _s(case_id: str, tone: int, worldview: int, memory: int, category: str = "greeting"):
    return CaseScore(
        case_id=case_id,
        category=category,
        scores={"tone": tone, "worldview": worldview, "memory": memory},
    )


class TestCommonSetOnly:
    def test_uses_intersection(self):
        before = [_s("a", 5, 5, 5), _s("b", 1, 1, 1)]
        after = [_s("a", 3, 3, 3), _s("c", 5, 5, 5)]

        pc = paired_compare(before, after)

        assert pc.common_ids == ["a"]
        assert pc.before.case_count == 1
        assert pc.after.case_count == 1

    def test_reports_dropped_cases(self):
        """무엇이 비교에서 빠졌는지 드러나야 한다."""
        before = [_s("a", 5, 5, 5), _s("b", 4, 4, 4)]
        after = [_s("a", 3, 3, 3), _s("c", 2, 2, 2)]

        pc = paired_compare(before, after)

        assert pc.dropped_before == ["b"]
        assert pc.dropped_after == ["c"]

    def test_unequal_sample_does_not_skew_delta(self):
        """한쪽에만 있는 고득점 사례가 평균을 밀어올리면 안 된다.

        before에만 있는 만점 사례를 넣어도, 공통 사례의 차이는 변하지 않아야 한다.
        """
        before = [_s("a", 3, 3, 3)]
        after = [_s("a", 4, 4, 4)]
        baseline = {d.axis: d.delta for d in paired_compare(before, after).deltas}

        before_skewed = [*before, _s("only-before", 5, 5, 5)]
        skewed = {d.axis: d.delta for d in paired_compare(before_skewed, after).deltas}

        assert skewed == baseline

    def test_no_overlap_yields_empty(self):
        pc = paired_compare([_s("a", 5, 5, 5)], [_s("b", 1, 1, 1)])

        assert pc.common_ids == []
        assert pc.before.case_count == 0


class TestCaseWins:
    def test_counts_wins_losses_ties(self):
        before = [_s("a", 3, 3, 3), _s("b", 5, 5, 5), _s("c", 4, 4, 4)]
        after = [_s("a", 4, 4, 4), _s("b", 2, 2, 2), _s("c", 4, 4, 4)]

        pc = paired_compare(before, after)

        assert (pc.wins_after, pc.wins_before, pc.ties) == (1, 1, 1)

    def test_wins_use_total_not_per_axis(self):
        """한 축이 내려도 총합이 오르면 우세로 센다."""
        before = [_s("a", 3, 3, 3)]
        after = [_s("a", 5, 5, 1)]

        pc = paired_compare(before, after)

        assert pc.wins_after == 1


class TestSerialization:
    def test_to_dict_exposes_sample_info(self):
        pc = paired_compare([_s("a", 3, 3, 3), _s("b", 3, 3, 3)], [_s("a", 4, 4, 4)])
        payload = pc.to_dict()

        assert payload["common_case_count"] == 1
        assert payload["dropped_from_before"] == ["b"]
        assert payload["deltas"]["overall"] == 1.0
        assert payload["case_wins"] == {"after": 1, "before": 0, "tie": 0}
