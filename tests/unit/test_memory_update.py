"""기억 갱신과 충돌 해소의 현재 동작 고정 (TASK-14, REQ-14-1).

`MemoryModule.update()`는 163줄이고 충돌 해소 분기가 셋인데, 착수 시점에
이 동작을 직접 검증하는 테스트가 **0개**였다. 파이프라인 통합 테스트가
간접적으로 지나갈 뿐 의미를 고정하는 단언이 없었다.

이 파일은 **특성화 테스트**다. 지금 동작이 옳다고 주장하지 않는다.
리팩터링이 동작을 바꾸지 않았음을 증명하는 것이 목적이다.

고정하는 동작:

| 분류 | 결과 |
|---|---|
| DIFFERENT | 새 기억 추가. `weight`는 importance, `emotion_tags`는 갱신 시점 감정 |
| IDENTICAL | 기존 기억 유지. `access_count` +1, `last_accessed` 갱신 |
| SIMILAR | 기존 기억의 content를 새 것으로 교체, `weight`는 **max** |

LLM API 키 없이 결정론적으로 통과한다.
"""

from __future__ import annotations

import numpy as np

from src.analysis import MemoryCandidate
from src.modules.memory import MemoryEntry, MemoryModule

# 임베딩을 테스트가 직접 지배해야 유사도 분기를 결정론적으로 고를 수 있다.
# 같은 그룹의 텍스트는 동일 벡터를 받아 유사도 1.0이 되고, 다른 그룹은 직교한다.
_VECTORS = {
    "A": np.array([1.0, 0.0], dtype=np.float32),
    "B": np.array([0.0, 1.0], dtype=np.float32),
}


def _embedding(text: str, _kind: str = "passage") -> np.ndarray:
    return _VECTORS["A"] if text.startswith("A:") else _VECTORS["B"]


class StubExtractor:
    """정해진 후보를 돌려주는 스텁. **LLM 더블이 아니다** — REQ-14-4."""

    def __init__(self, candidates: list[MemoryCandidate]):
        self._candidates = candidates

    def extract(self, user_input, character_response, history_context=""):
        return list(self._candidates)


class StubClassifier:
    """정해진 분류를 돌려주는 스텁. 호출 여부를 기록한다."""

    def __init__(self, classification: str = "DIFFERENT"):
        self._classification = classification
        self.calls: list[tuple[str, str]] = []

    def classify(self, existing_content: str, content: str) -> str:
        self.calls.append((existing_content, content))
        return self._classification


def _module(tmp_path) -> MemoryModule:
    return MemoryModule(db_path=str(tmp_path / "m.db"), embedding_fn=_embedding)


def _seed(module: MemoryModule, content: str, **kwargs) -> MemoryEntry:
    entry = MemoryEntry(
        id=f"seed-{content}",
        content=content,
        embedding=_embedding(content),
        weight=kwargs.get("weight", 0.5),
        access_count=kwargs.get("access_count", 0),
        last_accessed=kwargs.get("last_accessed", 0.0),
        created_at=0.0,
    )
    module._memories[entry.id] = entry
    return entry


# ---------------------------------------------------------------------------
# DIFFERENT — 새 기억 추가
# ---------------------------------------------------------------------------


class TestDifferent:
    def test_adds_new_memory_when_store_is_empty(self, tmp_path):
        module = _module(tmp_path)
        extractor, classifier = (
            StubExtractor([MemoryCandidate("A:사용자는 시험이 있다", 0.8)]),
            StubClassifier(),
        )

        module.update("입력", "응답", {}, extractor, classifier)

        assert module.snapshot_count() == 1
        entry = next(iter(module._memories.values()))
        assert entry.content == "A:사용자는 시험이 있다"
        assert entry.weight == 0.8

    def test_records_emotions_at_time_of_storage(self, tmp_path):
        module = _module(tmp_path)
        extractor, classifier = StubExtractor([MemoryCandidate("A:내용", 0.5)]), StubClassifier()

        module.update("입력", "응답", {"공감": 0.7}, extractor, classifier)

        entry = next(iter(module._memories.values()))
        assert entry.emotion_tags == {"공감": 0.7}

    def test_dissimilar_memory_is_added_without_asking_llm(self, tmp_path):
        """유사도가 임계값 미만이면 충돌 판정 호출 자체가 없어야 한다."""
        module = _module(tmp_path)
        _seed(module, "A:기존")
        extractor, classifier = (
            StubExtractor([MemoryCandidate("B:전혀 다른 내용", 0.5)]),
            StubClassifier(),
        )

        module.update("입력", "응답", {}, extractor, classifier)

        assert module.snapshot_count() == 2
        assert classifier.calls == [], "직교하는 기억에 판정 호출은 낭비다"

    def test_default_importance_is_applied(self, tmp_path):
        module = _module(tmp_path)
        extractor, classifier = StubExtractor([MemoryCandidate("A:내용")]), StubClassifier()

        module.update("입력", "응답", {}, extractor, classifier)

        assert next(iter(module._memories.values())).weight == 0.5

    def test_empty_content_is_skipped(self, tmp_path):
        module = _module(tmp_path)
        extractor, classifier = StubExtractor([MemoryCandidate("", 0.9)]), StubClassifier()

        module.update("입력", "응답", {}, extractor, classifier)

        assert module.snapshot_count() == 0


# ---------------------------------------------------------------------------
# IDENTICAL — 기존 기억 유지, 접근 기록만 갱신
# ---------------------------------------------------------------------------


class TestIdentical:
    def test_does_not_add_new_memory(self, tmp_path):
        module = _module(tmp_path)
        _seed(module, "A:기존 내용")
        extractor, classifier = (
            StubExtractor([MemoryCandidate("A:같은 내용", 0.9)]),
            StubClassifier("IDENTICAL"),
        )

        module.update("입력", "응답", {}, extractor, classifier)

        assert module.snapshot_count() == 1

    def test_keeps_original_content(self, tmp_path):
        module = _module(tmp_path)
        _seed(module, "A:기존 내용")
        extractor, classifier = (
            StubExtractor([MemoryCandidate("A:같은 내용", 0.9)]),
            StubClassifier("IDENTICAL"),
        )

        module.update("입력", "응답", {}, extractor, classifier)

        assert module._memories["seed-A:기존 내용"].content == "A:기존 내용"

    def test_increments_access_count(self, tmp_path):
        module = _module(tmp_path)
        _seed(module, "A:기존 내용", access_count=3)
        extractor, classifier = (
            StubExtractor([MemoryCandidate("A:같은 내용")]),
            StubClassifier("IDENTICAL"),
        )

        module.update("입력", "응답", {}, extractor, classifier)

        assert module._memories["seed-A:기존 내용"].access_count == 4

    def test_does_not_change_weight(self, tmp_path):
        module = _module(tmp_path)
        _seed(module, "A:기존 내용", weight=0.2)
        extractor, classifier = (
            StubExtractor([MemoryCandidate("A:같은 내용", 0.9)]),
            StubClassifier("IDENTICAL"),
        )

        module.update("입력", "응답", {}, extractor, classifier)

        assert module._memories["seed-A:기존 내용"].weight == 0.2


# ---------------------------------------------------------------------------
# SIMILAR — 내용 교체, 가중치는 최대값 유지
# ---------------------------------------------------------------------------


class TestSimilar:
    def test_replaces_content(self, tmp_path):
        module = _module(tmp_path)
        _seed(module, "A:오래된 내용")
        extractor, classifier = (
            StubExtractor([MemoryCandidate("A:새 내용")]),
            StubClassifier("SIMILAR"),
        )

        module.update("입력", "응답", {}, extractor, classifier)

        assert module._memories["seed-A:오래된 내용"].content == "A:새 내용"

    def test_does_not_add_new_memory(self, tmp_path):
        module = _module(tmp_path)
        _seed(module, "A:오래된 내용")
        extractor, classifier = (
            StubExtractor([MemoryCandidate("A:새 내용")]),
            StubClassifier("SIMILAR"),
        )

        module.update("입력", "응답", {}, extractor, classifier)

        assert module.snapshot_count() == 1

    def test_weight_takes_the_maximum(self, tmp_path):
        module = _module(tmp_path)
        _seed(module, "A:오래된 내용", weight=0.9)
        extractor, classifier = (
            StubExtractor([MemoryCandidate("A:새 내용", 0.1)]),
            StubClassifier("SIMILAR"),
        )

        module.update("입력", "응답", {}, extractor, classifier)

        assert module._memories["seed-A:오래된 내용"].weight == 0.9, "병합은 가중치를 낮추지 않는다"

    def test_weight_is_raised_when_new_is_higher(self, tmp_path):
        module = _module(tmp_path)
        _seed(module, "A:오래된 내용", weight=0.1)
        extractor, classifier = (
            StubExtractor([MemoryCandidate("A:새 내용", 0.9)]),
            StubClassifier("SIMILAR"),
        )

        module.update("입력", "응답", {}, extractor, classifier)

        assert module._memories["seed-A:오래된 내용"].weight == 0.9


# ---------------------------------------------------------------------------
# 응답 처리
# ---------------------------------------------------------------------------


class TestExtractionResults:
    """추출 결과가 비어 있는 경우.

    거부·파싱 실패를 빈 목록으로 바꾸는 일은 이제 분석 층의 책임이며
    `test_analysis.py`가 검증한다. 도메인은 "빈 목록이면 아무것도 안 한다"만 안다.
    """

    def test_no_candidates_stores_nothing(self, tmp_path):
        module = _module(tmp_path)
        module.update("입력", "응답", {}, StubExtractor([]), StubClassifier())

        assert module.snapshot_count() == 0

    def test_multiple_candidates_are_all_processed(self, tmp_path):
        module = _module(tmp_path)
        extractor = StubExtractor([MemoryCandidate("A:첫째"), MemoryCandidate("B:둘째")])

        module.update("입력", "응답", {}, extractor, StubClassifier("DIFFERENT"))

        assert module.snapshot_count() == 2
