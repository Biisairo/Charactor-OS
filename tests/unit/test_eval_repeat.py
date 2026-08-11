"""반복 실행 평균과 변동 폭 (TASK-01 후속 / REQ-08-7).

1회 실행의 점수는 판정자 변동으로 흔들린다. 실제로 같은 설정을 두 번 돌렸을 때
전체 평균이 0.49 차이 났고, 그 탓에 설정 간 차이(0.06~0.29)를 판별할 수 없었다.
반복 평균으로 변동을 줄이고, 변동 폭 자체를 노이즈 추정치로 보고한다.
"""

from __future__ import annotations

from eval.scoring import CaseScore, Summary, average_across_runs, run_spread


def _s(case_id: str, value: float, category: str = "greeting") -> CaseScore:
    return CaseScore(
        case_id=case_id,
        category=category,
        scores={"tone": value, "worldview": value, "memory": value},
    )


class TestAverageAcrossRuns:
    def test_averages_per_case(self):
        runs = [[_s("a", 5), _s("b", 1)], [_s("a", 3), _s("b", 3)]]

        averaged = {cs.case_id: cs.scores["tone"] for cs in average_across_runs(runs)}

        assert averaged == {"a": 4.0, "b": 2.0}

    def test_single_run_passes_through(self):
        averaged = average_across_runs([[_s("a", 4)]])

        assert averaged[0].scores["tone"] == 4.0

    def test_drops_cases_missing_from_any_run(self):
        """일부 실행에서만 채점된 사례를 섞으면 실행마다 표본이 달라진다."""
        runs = [[_s("a", 5), _s("b", 5)], [_s("a", 3)]]

        ids = [cs.case_id for cs in average_across_runs(runs)]

        assert ids == ["a"]

    def test_preserves_category(self):
        runs = [[_s("a", 5, "worldview")], [_s("a", 3, "worldview")]]

        assert average_across_runs(runs)[0].category == "worldview"

    def test_empty(self):
        assert average_across_runs([]) == []

    def test_three_runs(self):
        runs = [[_s("a", 5)], [_s("a", 4)], [_s("a", 3)]]

        assert average_across_runs(runs)[0].scores["tone"] == 4.0


class TestRunSpread:
    def _summary(self, overall: float) -> Summary:
        return Summary(per_axis={"tone": overall}, overall=overall, case_count=1)

    def test_reports_range(self):
        spread = run_spread([self._summary(4.2), self._summary(4.7), self._summary(4.3)])

        assert spread["runs"] == 3
        assert spread["min"] == 4.2
        assert spread["max"] == 4.7
        assert spread["spread"] == 0.5

    def test_mean(self):
        spread = run_spread([self._summary(4.0), self._summary(5.0)])

        assert spread["mean"] == 4.5

    def test_single_run_has_no_spread(self):
        """1회 실행으로는 노이즈를 추정할 수 없다."""
        spread = run_spread([self._summary(4.3)])

        assert spread["runs"] == 1
        assert spread["spread"] == 0.0

    def test_identical_runs(self):
        spread = run_spread([self._summary(4.3), self._summary(4.3)])

        assert spread["spread"] == 0.0
