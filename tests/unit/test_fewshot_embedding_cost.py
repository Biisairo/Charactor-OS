"""FewShot 검색의 임베딩 계산 횟수 (SPEC-12 REQ-21-17 ~ 21-19).

검색 한 번에 질의 1회 + **예시마다 1회** 임베딩을 계산하고 있었다(P-5).
예시가 50개면 51회다. 모델이 5배 무거워지면 그대로 5배 느려지는 경로이므로,
이 테스트가 보는 것은 검색 품질이 아니라 **계산 횟수**다.
"""

from __future__ import annotations

import numpy as np
import yaml

from src.embedding import PASSAGE, QUERY
from src.modules.fewshot import FewShotModule


class _Counter:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, text: str, kind: str) -> np.ndarray:
        self.calls.append((text, kind))
        return np.array([1.0, 0.0], dtype=np.float32)

    def kinds(self, kind: str) -> int:
        return sum(1 for _text, k in self.calls if k == kind)


def _write_examples(directory, tag: str, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "examples": [{"user": f"질문 {i}", "character": f"응답 {i}"} for i in range(count)],
    }
    (directory / f"{tag}.yaml").write_text(yaml.dump(payload, allow_unicode=True), encoding="utf-8")


def _module(tmp_path, counter, *, count: int = 5) -> FewShotModule:
    _write_examples(tmp_path, "일상", count)
    module = FewShotModule(str(tmp_path), embedding_fn=counter)
    module.load_all()
    return module


class TestExamplesAreEmbeddedOnce:
    def test_examples_are_embedded_at_load(self, tmp_path):
        counter = _Counter()
        _module(tmp_path, counter, count=5)

        assert counter.kinds(PASSAGE) == 5

    def test_search_does_not_embed_examples(self, tmp_path):
        counter = _Counter()
        module = _module(tmp_path, counter, count=5)
        counter.calls.clear()

        module.search("뭐해")

        assert counter.kinds(PASSAGE) == 0

    def test_search_embeds_query_once(self, tmp_path):
        """예시 루프 안에서 질의를 재계산하던 자리다."""
        counter = _Counter()
        module = _module(tmp_path, counter, count=5)
        counter.calls.clear()

        module.search("뭐해")

        assert counter.kinds(QUERY) == 1

    def test_repeated_search_does_not_reembed_examples(self, tmp_path):
        counter = _Counter()
        module = _module(tmp_path, counter, count=5)
        counter.calls.clear()

        module.search("뭐해")
        module.search("밥 먹었어")

        assert counter.kinds(PASSAGE) == 0
        assert counter.kinds(QUERY) == 2

    def test_cost_does_not_grow_with_example_count(self, tmp_path):
        """예시를 늘려도 검색 비용은 그대로여야 한다."""
        counter = _Counter()
        module = _module(tmp_path, counter, count=40)
        counter.calls.clear()

        module.search("뭐해")

        assert len(counter.calls) == 1

    def test_runtime_added_example_is_embedded(self, tmp_path):
        """런타임 추가 예시가 임베딩 없이 남으면 검색에서 조용히 밀린다."""
        counter = _Counter()
        module = _module(tmp_path, counter, count=1)
        counter.calls.clear()

        module.add_example("일상", "새 질문", "새 응답")

        assert counter.kinds(PASSAGE) == 1


class TestFallbackIsPreserved:
    def test_no_embedding_fn_still_searches(self, tmp_path):
        """임베딩 없는 폴백(태그 0.7 + 감정 0.3)은 그대로 유지된다 (REQ-21-19)."""
        _write_examples(tmp_path, "인사", 1)
        module = FewShotModule(str(tmp_path))
        module.load_all()

        assert module.search("안녕")

    def test_broken_embedding_degrades_and_reports(self, tmp_path):
        def broken(_text: str, _kind: str):
            raise RuntimeError("모델 없음")

        _write_examples(tmp_path, "인사", 1)
        module = FewShotModule(str(tmp_path), embedding_fn=broken)
        module.load_all()

        assert module.search("안녕") is not None
        assert any(not issue.expected for issue in module.load_issues)
