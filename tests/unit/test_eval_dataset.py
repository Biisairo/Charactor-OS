"""골든 데이터셋 검증 (TASK-01 / REQ-01-1, 01-2, 01-8).

데이터셋은 사람이 손으로 쓰는 YAML이므로, 잘못된 항목이 조용히 통과하면
평가 결과가 왜곡된다. 로드 시점에 강하게 검증한다.
"""

from __future__ import annotations

import pytest

from eval.dataset import (
    MIN_CASES_PER_CATEGORY,
    MIN_TOTAL_CASES,
    REQUIRED_CATEGORIES,
    DatasetError,
    load_dataset,
    parse_dataset,
    validate_coverage,
)


def _case(case_id: str, category: str = "greeting", **overrides) -> dict:
    base = {
        "id": case_id,
        "category": category,
        "input": "안녕하시오",
        "expectation": "말투를 유지한다",
    }
    base.update(overrides)
    return base


def _payload(cases: list[dict]) -> dict:
    return {"character": "test-char", "description": "테스트", "cases": cases}


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------


class TestParseDataset:
    def test_minimal_valid(self):
        dataset = parse_dataset(_payload([_case("a")]))

        assert dataset.character == "test-char"
        assert dataset.cases[0].id == "a"
        assert dataset.cases[0].setup == []

    def test_setup_is_preserved(self):
        dataset = parse_dataset(
            _payload([_case("a", setup=["내 이름은 박서준이오", "한양에 사오"])])
        )

        assert dataset.cases[0].setup == ["내 이름은 박서준이오", "한양에 사오"]

    def test_categories_count(self):
        dataset = parse_dataset(_payload([_case("a"), _case("b"), _case("c", "worldview")]))

        assert dataset.categories() == {"greeting": 2, "worldview": 1}

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({}, "character"),
            ({"character": "x"}, "cases"),
            ({"character": "x", "cases": []}, "cases"),
        ],
    )
    def test_structural_errors(self, payload: dict, match: str):
        with pytest.raises(DatasetError, match=match):
            parse_dataset(payload)

    def test_duplicate_id_rejected(self):
        with pytest.raises(DatasetError, match="중복된 사례 id"):
            parse_dataset(_payload([_case("dup"), _case("dup")]))

    def test_unknown_category_rejected(self):
        with pytest.raises(DatasetError, match="알 수 없는 범주"):
            parse_dataset(_payload([_case("a", "무작위범주")]))

    def test_empty_input_rejected(self):
        with pytest.raises(DatasetError, match="'input'이 비어"):
            parse_dataset(_payload([_case("a", input="   ")]))

    def test_missing_expectation_rejected(self):
        """기대 동작이 없으면 판정자가 무엇을 기준으로 채점할지 알 수 없다."""
        with pytest.raises(DatasetError, match="'expectation'이 비어"):
            parse_dataset(_payload([_case("a", expectation="")]))

    def test_setup_must_be_list(self):
        with pytest.raises(DatasetError, match="'setup'은 리스트"):
            parse_dataset(_payload([_case("a", setup="문자열")]))


# ---------------------------------------------------------------------------
# 커버리지 요건 (REQ-01-2)
# ---------------------------------------------------------------------------


class TestValidateCoverage:
    def test_insufficient_category_rejected(self):
        cases = [_case(f"g{i}", "greeting") for i in range(MIN_TOTAL_CASES)]
        with pytest.raises(DatasetError, match="범주당 최소"):
            validate_coverage(parse_dataset(_payload(cases)))

    def test_insufficient_total_rejected(self):
        """모든 범주를 최소치만 채우면 총량 요건에 걸린다."""
        cases = [
            _case(f"{cat}-{i}", cat)
            for cat in REQUIRED_CATEGORIES
            for i in range(MIN_CASES_PER_CATEGORY)
        ]
        assert len(cases) < MIN_TOTAL_CASES

        with pytest.raises(DatasetError, match="최소 20건"):
            validate_coverage(parse_dataset(_payload(cases)))

    def test_satisfied(self):
        cases = [_case(f"{cat}-{i}", cat) for cat in REQUIRED_CATEGORIES for i in range(4)]
        validate_coverage(parse_dataset(_payload(cases)))  # 예외 없음


# ---------------------------------------------------------------------------
# 실제 데이터셋 — 리포지토리에 담긴 파일이 요건을 만족해야 한다
# ---------------------------------------------------------------------------


class TestShippedDataset:
    def test_hong_gil_dong_loads_and_satisfies_coverage(self):
        dataset = load_dataset("hong-gil-dong")

        assert len(dataset.cases) >= MIN_TOTAL_CASES
        for category in REQUIRED_CATEGORIES:
            assert dataset.categories().get(category, 0) >= MIN_CASES_PER_CATEGORY

    def test_memory_cases_have_setup(self):
        """기억 참조 사례는 참조할 기억이 없으면 성립하지 않는다."""
        dataset = load_dataset("hong-gil-dong")

        memory_cases = [c for c in dataset.cases if c.category == "memory_recall"]
        assert memory_cases
        for case in memory_cases:
            assert case.setup, f"{case.id}: setup 발화가 필요하다"

    def test_missing_dataset_reports_available(self):
        with pytest.raises(DatasetError, match="사용 가능"):
            load_dataset("존재하지-않는-캐릭터")
