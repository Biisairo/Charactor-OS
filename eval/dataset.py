"""골든 데이터셋 로드와 구성 검증 (REQ-01-1, REQ-01-2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DATASET_DIR = Path(__file__).parent / "datasets"

# REQ-01-2 — 필수 범주와 범주별 최소 사례 수
REQUIRED_CATEGORIES = (
    "greeting",  # 인사
    "emotional_appeal",  # 감정 호소
    "worldview",  # 세계관 질문
    "memory_recall",  # 기억 참조
    "persona_break",  # 페르소나 이탈 유도
)
MIN_CASES_PER_CATEGORY = 3
MIN_TOTAL_CASES = 20


class DatasetError(ValueError):
    """데이터셋이 요구 조건을 만족하지 않을 때 발생한다."""


@dataclass(frozen=True)
class GoldenCase:
    """평가 입력 1건.

    setup: 이 사례를 평가하기 전에 먼저 대화해 둘 사용자 발화들.
        기억 참조 사례는 참조할 기억이 있어야 성립하므로 필요하다.
    expectation: 판정자에게 전달할 기대 동작 서술.
    """

    id: str
    category: str
    input: str
    expectation: str
    setup: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GoldenDataset:
    character: str
    description: str
    cases: list[GoldenCase]

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.category] = counts.get(case.category, 0) + 1
        return counts


def dataset_path(character: str) -> Path:
    return DATASET_DIR / f"{character}.yaml"


def parse_dataset(payload: dict) -> GoldenDataset:
    """dict에서 데이터셋을 만든다. 파일 접근이 없어 단위 테스트가 쉽다."""
    if not isinstance(payload, dict):
        raise DatasetError("데이터셋 최상위는 매핑이어야 합니다")

    character = str(payload.get("character") or "").strip()
    if not character:
        raise DatasetError("'character' 필드가 필요합니다")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise DatasetError("'cases'는 비어 있지 않은 리스트여야 합니다")

    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise DatasetError(f"cases[{index}]가 매핑이 아닙니다")

        case_id = str(raw.get("id") or "").strip()
        if not case_id:
            raise DatasetError(f"cases[{index}]에 'id'가 없습니다")
        if case_id in seen:
            raise DatasetError(f"중복된 사례 id: {case_id}")
        seen.add(case_id)

        category = str(raw.get("category") or "").strip()
        if category not in REQUIRED_CATEGORIES:
            raise DatasetError(
                f"{case_id}: 알 수 없는 범주 '{category}' (허용: {', '.join(REQUIRED_CATEGORIES)})"
            )

        text = str(raw.get("input") or "").strip()
        if not text:
            raise DatasetError(f"{case_id}: 'input'이 비어 있습니다")

        expectation = str(raw.get("expectation") or "").strip()
        if not expectation:
            raise DatasetError(f"{case_id}: 'expectation'이 비어 있습니다")

        setup_raw = raw.get("setup") or []
        if not isinstance(setup_raw, list):
            raise DatasetError(f"{case_id}: 'setup'은 리스트여야 합니다")

        cases.append(
            GoldenCase(
                id=case_id,
                category=category,
                input=text,
                expectation=expectation,
                setup=[str(s) for s in setup_raw],
            )
        )

    return GoldenDataset(
        character=character,
        description=str(payload.get("description") or "").strip(),
        cases=cases,
    )


def validate_coverage(dataset: GoldenDataset) -> None:
    """REQ-01-2의 범주·개수 요건을 검사한다."""
    counts = dataset.categories()

    missing = [c for c in REQUIRED_CATEGORIES if counts.get(c, 0) < MIN_CASES_PER_CATEGORY]
    if missing:
        detail = ", ".join(f"{c}({counts.get(c, 0)}건)" for c in missing)
        raise DatasetError(f"범주당 최소 {MIN_CASES_PER_CATEGORY}건이 필요합니다 — 부족: {detail}")

    if len(dataset.cases) < MIN_TOTAL_CASES:
        raise DatasetError(
            f"사례가 최소 {MIN_TOTAL_CASES}건 필요합니다 (현재 {len(dataset.cases)}건)"
        )


def load_dataset(character: str, *, validate: bool = True) -> GoldenDataset:
    path = dataset_path(character)
    if not path.exists():
        available = sorted(p.stem for p in DATASET_DIR.glob("*.yaml"))
        raise DatasetError(
            f"데이터셋이 없습니다: {path}\n사용 가능: {', '.join(available) or '(없음)'}"
        )

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    dataset = parse_dataset(payload)
    if validate:
        validate_coverage(dataset)
    return dataset
