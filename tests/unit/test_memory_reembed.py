"""모델 교체 시 기억 재임베딩 (SPEC-12 REQ-21-20 ~ 21-23).

기억 벡터는 SQLite에 남는다. 임베딩 모델을 바꾸면 좌표계가 달라지므로, 낡은
벡터와 새 질의 벡터를 비교하면 **오류 없이 무의미한 점수**가 나온다(P-7).
그래서 DB 가 자기 벡터를 무엇으로 만들었는지 기록해야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.modules.memory import MemoryEntry, MemoryModule

MODEL_A = "test/model-a"
MODEL_B = "test/model-b"


def _vector(seed: int) -> np.ndarray:
    """모델별로 다른 좌표계를 흉내낸다."""
    vec = np.zeros(4, dtype=np.float32)
    vec[seed % 4] = 1.0
    return vec


class _Embedder:
    """호출을 세는 임베딩 더블. 모델마다 다른 축을 쓴다."""

    def __init__(self, axis: int):
        self._axis = axis
        self.calls: list[tuple[str, str]] = []

    def __call__(self, text: str, kind: str) -> np.ndarray:
        self.calls.append((text, kind))
        return _vector(self._axis)

    @property
    def count(self) -> int:
        return len(self.calls)


def _module(tmp_path, embedder, *, model: str) -> MemoryModule:
    return MemoryModule(db_path=str(tmp_path / "m.db"), embedding_fn=embedder, model_id=model)


def _seed(tmp_path, embedder, *, model: str, count: int = 3) -> None:
    """기억이 이미 쌓인 DB 를 만든다."""
    module = _module(tmp_path, embedder, model=model)
    for i in range(count):
        module._memories[f"m{i}"] = MemoryEntry(
            id=f"m{i}",
            content=f"기억 {i}",
            embedding=embedder(f"기억 {i}", "passage"),
        )
    module.save()


# ---------------------------------------------------------------------------
# 1. 모델 식별자 기록 (REQ-21-20)
# ---------------------------------------------------------------------------


class TestModelIsRecorded:
    def test_same_model_does_not_recompute(self, tmp_path):
        _seed(tmp_path, _Embedder(0), model=MODEL_A)

        fresh = _Embedder(0)
        _module(tmp_path, fresh, model=MODEL_A).load()

        assert fresh.count == 0

    def test_missing_db_is_not_an_error(self, tmp_path):
        fresh = _Embedder(0)
        _module(tmp_path, fresh, model=MODEL_A).load()

        assert fresh.count == 0


# ---------------------------------------------------------------------------
# 2. 불일치 시 재임베딩 (REQ-21-21)
# ---------------------------------------------------------------------------


class TestReembedOnModelChange:
    def test_model_change_reembeds_everything(self, tmp_path):
        _seed(tmp_path, _Embedder(0), model=MODEL_A, count=3)

        new = _Embedder(1)
        _module(tmp_path, new, model=MODEL_B).load()

        assert new.count == 3

    def test_reembedded_vectors_are_in_the_new_space(self, tmp_path):
        _seed(tmp_path, _Embedder(0), model=MODEL_A, count=1)

        new = _Embedder(1)
        module = _module(tmp_path, new, model=MODEL_B)
        module.load()

        np.testing.assert_allclose(module._memories["m0"].embedding, _vector(1))

    def test_reembedding_is_persisted(self, tmp_path):
        """재계산 결과를 저장하지 않으면 부팅마다 다시 계산한다."""
        _seed(tmp_path, _Embedder(0), model=MODEL_A, count=2)
        _module(tmp_path, _Embedder(1), model=MODEL_B).load()

        third = _Embedder(1)
        _module(tmp_path, third, model=MODEL_B).load()

        assert third.count == 0

    def test_content_is_the_source_of_truth(self, tmp_path):
        """원문이 DB 에 있으므로 재계산이 가능하다."""
        _seed(tmp_path, _Embedder(0), model=MODEL_A, count=2)

        new = _Embedder(1)
        _module(tmp_path, new, model=MODEL_B).load()

        assert {text for text, _kind in new.calls} == {"기억 0", "기억 1"}

    def test_reembedding_uses_passage_kind(self, tmp_path):
        """기억 내용은 찾히는 쪽이다 (REQ-21-4)."""
        _seed(tmp_path, _Embedder(0), model=MODEL_A, count=1)

        new = _Embedder(1)
        _module(tmp_path, new, model=MODEL_B).load()

        assert new.calls[0][1] == "passage"

    def test_count_is_reported(self, tmp_path):
        """조용히 일어나서는 안 된다 (REQ-21-22)."""
        _seed(tmp_path, _Embedder(0), model=MODEL_A, count=3)

        module = _module(tmp_path, _Embedder(1), model=MODEL_B)
        module.load()

        assert module.reembedded == 3

    def test_no_reembedding_reports_zero(self, tmp_path):
        _seed(tmp_path, _Embedder(0), model=MODEL_A, count=3)

        module = _module(tmp_path, _Embedder(0), model=MODEL_A)
        module.load()

        assert module.reembedded == 0


# ---------------------------------------------------------------------------
# 3. 기록이 없는 기존 DB (REQ-21-20)
# ---------------------------------------------------------------------------


class TestLegacyDatabase:
    def test_db_without_model_is_treated_as_stale(self, tmp_path):
        """식별자가 없는 DB 는 무엇으로 만들었는지 알 수 없다. 낡은 것으로 본다."""
        _seed(tmp_path, _Embedder(0), model="")

        new = _Embedder(1)
        _module(tmp_path, new, model=MODEL_A).load()

        assert new.count == 3


# ---------------------------------------------------------------------------
# 4. 중단 안전성 (리스크 HIGH)
# ---------------------------------------------------------------------------


class TestInterruption:
    def test_failed_reembedding_is_retried_next_load(self, tmp_path):
        """식별자를 재임베딩 **완료 후**에 커밋해야 중단이 안전하다."""
        _seed(tmp_path, _Embedder(0), model=MODEL_A, count=2)

        class _Broken:
            def __call__(self, _text: str, _kind: str):
                raise RuntimeError("모델 로드 실패")

        _module(tmp_path, _Broken(), model=MODEL_B).load()

        retry = _Embedder(1)
        _module(tmp_path, retry, model=MODEL_B).load()

        assert retry.count == 2

    def test_failed_memories_are_excluded_from_search(self, tmp_path):
        """좌표계가 다른 벡터로 검색을 계속하면 점수가 조용히 무의미해진다."""
        _seed(tmp_path, _Embedder(0), model=MODEL_A, count=2)

        class _Broken:
            def __call__(self, _text: str, _kind: str):
                raise RuntimeError("모델 로드 실패")

        module = _module(tmp_path, _Broken(), model=MODEL_B)
        module.load()

        assert module._memories == {}

    def test_failure_is_reported(self, tmp_path):
        _seed(tmp_path, _Embedder(0), model=MODEL_A, count=1)

        class _Broken:
            def __call__(self, _text: str, _kind: str):
                raise RuntimeError("모델 로드 실패")

        module = _module(tmp_path, _Broken(), model=MODEL_B)
        module.load()

        assert any(not issue.expected for issue in module.load_issues)


# ---------------------------------------------------------------------------
# 5. 검색 질의의 용도 (REQ-21-4)
# ---------------------------------------------------------------------------


class TestSearchKind:
    @pytest.mark.parametrize("count", [0, 2])
    def test_query_uses_query_kind(self, tmp_path, count):
        embedder = _Embedder(0)
        _seed(tmp_path, embedder, model=MODEL_A, count=count)

        module = _module(tmp_path, embedder, model=MODEL_A)
        module.load()
        embedder.calls.clear()
        module.search("무엇을 기억하니")

        assert all(kind == "query" for _text, kind in embedder.calls)
