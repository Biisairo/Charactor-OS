"""평가 결과 저장 검증 (TASK-01 / REQ-01-4, REQ-01-5).

결과 파일명이 고정이면 재실행이 이전 결과를 조용히 덮어쓴다. 실제로 그렇게
해서 어느 실행의 수치인지 추적할 수 없게 된 적이 있어 회귀 테스트로 고정한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.dataset import GoldenCase
from eval.runner import CaseResult, EvalRun
from eval.scoring import CaseScore


def _run(setting: str, timestamp: str, tone: int = 5) -> EvalRun:
    case = GoldenCase(id="c1", category="greeting", input="안녕", expectation="인사한다")
    return EvalRun(
        character="test-char",
        setting=setting,
        target_model="target-model",
        judge_model="judge-model",
        commit="abc1234",
        timestamp=timestamp,
        results=[
            CaseResult(
                case=case,
                response="반갑소",
                score=CaseScore(
                    case_id="c1",
                    category="greeting",
                    scores={"tone": tone, "worldview": 4, "memory": 3},
                ),
            )
        ],
    )


class TestFilenameUniqueness:
    def test_different_timestamps_produce_different_files(self, tmp_path: Path):
        first = _run("reflection-on", "2026-08-11T14:43:33").save(tmp_path)
        second = _run("reflection-on", "2026-08-11T15:04:12").save(tmp_path)

        assert first != second
        assert first.exists() and second.exists()

    def test_rerun_does_not_overwrite_previous(self, tmp_path: Path):
        first = _run("reflection-on", "2026-08-11T14:43:33", tone=5).save(tmp_path)
        _run("reflection-on", "2026-08-11T15:04:12", tone=1).save(tmp_path)

        payload = json.loads(first.read_text(encoding="utf-8"))
        assert payload["cases"][0]["scores"]["tone"] == 5, "이전 실행 결과가 보존되어야 한다"

    def test_settings_do_not_collide(self, tmp_path: Path):
        on = _run("reflection-on", "2026-08-11T14:43:33").save(tmp_path)
        off = _run("reflection-off", "2026-08-11T14:43:33").save(tmp_path)

        assert on != off


class TestLatestPointer:
    def test_latest_is_written(self, tmp_path: Path):
        _run("reflection-on", "2026-08-11T14:43:33").save(tmp_path)

        latest = tmp_path / "test-char_reflection-on_latest.json"
        assert latest.exists()

    def test_latest_reflects_most_recent_run(self, tmp_path: Path):
        _run("reflection-on", "2026-08-11T14:43:33", tone=5).save(tmp_path)
        _run("reflection-on", "2026-08-11T15:04:12", tone=2).save(tmp_path)

        latest = json.loads(
            (tmp_path / "test-char_reflection-on_latest.json").read_text(encoding="utf-8")
        )
        assert latest["cases"][0]["scores"]["tone"] == 2


class TestProvenance:
    def test_records_models_commit_and_time(self, tmp_path: Path):
        """어떤 조건에서 얻은 수치인지 결과만 보고 알 수 있어야 한다 (REQ-01-5, 01-11)."""
        path = _run("reflection-on", "2026-08-11T14:43:33").save(tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["target_model"] == "target-model"
        assert payload["judge_model"] == "judge-model"
        assert payload["commit"] == "abc1234"
        assert payload["timestamp"] == "2026-08-11T14:43:33"

    def test_summary_matches_cases(self, tmp_path: Path):
        path = _run("reflection-on", "2026-08-11T14:43:33").save(tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["summary"]["per_axis"]["tone"] == 5.0
        assert payload["summary"]["case_count"] == 1

    def test_failed_cases_are_listed(self, tmp_path: Path):
        run = _run("reflection-on", "2026-08-11T14:43:33")
        run.results.append(
            CaseResult(
                case=GoldenCase(id="c2", category="greeting", input="x", expectation="y"),
                response="",
                score=None,
                error="판정 실패",
            )
        )
        payload = json.loads(run.save(tmp_path).read_text(encoding="utf-8"))

        assert payload["failed_cases"] == [{"id": "c2", "reason": "판정 실패"}]
        assert payload["excluded_count"] == 1
        assert payload["summary"]["case_count"] == 1, "실패 사례는 평균에서 제외된다"
