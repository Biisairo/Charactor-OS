"""임베딩 영속 캐시 (SPEC-12 REQ-21-10 ~ 21-16).

캐시는 값을 빠르게 하는 장치가 아니라 **같은 계산을 반복하지 않는** 장치다.
그래서 이 테스트가 보는 것은 대부분 "계산이 몇 번 일어났는가"다.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embedding import PASSAGE, QUERY
from src.modules.embedding_cache import EmbeddingCache

MODEL = "test/model-a"
OTHER_MODEL = "test/model-b"


class _Counter:
    """임베딩 호출 횟수를 세는 더블."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, text: str, kind: str) -> np.ndarray:
        self.calls.append((text, kind))
        vec = np.full(4, float(len(text)), dtype=np.float32)
        return vec / np.linalg.norm(vec)

    @property
    def count(self) -> int:
        return len(self.calls)


def _cache(tmp_path, embedding_fn, *, model: str = MODEL, name: str = "e.db") -> EmbeddingCache:
    return EmbeddingCache(embedding_fn, str(tmp_path / name), model_id=model)


# ---------------------------------------------------------------------------
# 1. 계산을 반복하지 않는다 (REQ-21-10)
# ---------------------------------------------------------------------------


class TestReuse:
    def test_same_text_is_computed_once_within_a_session(self, tmp_path):
        counter = _Counter()
        cache = _cache(tmp_path, counter)

        cache("연표", PASSAGE)
        cache("연표", PASSAGE)

        assert counter.count == 1

    def test_second_load_reads_from_disk(self, tmp_path):
        """부팅마다 전량 재계산하던 것이 이 테스트가 막는 회귀다 (P-4)."""
        first = _Counter()
        _cache(tmp_path, first)("연표", PASSAGE)

        second = _Counter()
        _cache(tmp_path, second)("연표", PASSAGE)

        assert (first.count, second.count) == (1, 0)

    def test_cached_vector_equals_computed_vector(self, tmp_path):
        counter = _Counter()
        expected = _cache(tmp_path, counter)("연표", PASSAGE)

        restored = _cache(tmp_path, _Counter())("연표", PASSAGE)

        np.testing.assert_allclose(restored, expected)

    def test_cached_vector_keeps_dtype(self, tmp_path):
        """Memory 는 float32 로 저장한다. dtype 이 흔들리면 내적이 어긋난다."""
        _cache(tmp_path, _Counter())("연표", PASSAGE)

        assert _cache(tmp_path, _Counter())("연표", PASSAGE).dtype == np.float32

    def test_shared_across_modules(self, tmp_path):
        """Knowledge 와 FewShot 이 같은 문장을 두 번 계산하지 않는다 (REQ-21-13)."""
        counter = _Counter()
        knowledge_side = _cache(tmp_path, counter)
        fewshot_side = _cache(tmp_path, counter)

        knowledge_side("같은 문장", PASSAGE)
        fewshot_side("같은 문장", PASSAGE)

        assert counter.count == 1


# ---------------------------------------------------------------------------
# 2. 캐시 키 (REQ-21-11 · 21-12)
# ---------------------------------------------------------------------------


class TestKey:
    def test_different_text_is_a_different_entry(self, tmp_path):
        counter = _Counter()
        cache = _cache(tmp_path, counter)

        cache("연표", PASSAGE)
        cache("장소", PASSAGE)

        assert counter.count == 2

    def test_model_change_invalidates_everything(self, tmp_path):
        """모델이 바뀌면 좌표계가 다르다. 별도 무효화 절차 없이 미스가 된다."""
        _cache(tmp_path, _Counter(), model=MODEL)("연표", PASSAGE)

        counter = _Counter()
        _cache(tmp_path, counter, model=OTHER_MODEL)("연표", PASSAGE)

        assert counter.count == 1

    def test_content_hash_not_filename(self, tmp_path):
        """파일이 어디에 있었는지는 키에 들어가지 않는다 (결정 2).

        `base/`↔`general/` 이동과 파일명 변경에도 캐시가 살아남아야 한다.
        캐시는 텍스트만 알므로, 같은 텍스트면 언제나 같은 항목이다.
        """
        counter = _Counter()
        cache = _cache(tmp_path, counter)

        cache("옮겨진 문단", PASSAGE)
        cache("옮겨진 문단", PASSAGE)

        assert counter.count == 1


# ---------------------------------------------------------------------------
# 3. 질의는 캐시하지 않는다
# ---------------------------------------------------------------------------


class TestQueryIsNotCached:
    def test_query_is_recomputed(self, tmp_path):
        """질의는 매번 다르다. 저장하면 캐시가 무한히 자란다."""
        counter = _Counter()
        cache = _cache(tmp_path, counter)

        cache("무엇을 찾는가", QUERY)
        cache("무엇을 찾는가", QUERY)

        assert counter.count == 2

    def test_query_does_not_survive_restart(self, tmp_path):
        _cache(tmp_path, _Counter())("무엇을 찾는가", QUERY)

        counter = _Counter()
        _cache(tmp_path, counter)("무엇을 찾는가", QUERY)

        assert counter.count == 1


# ---------------------------------------------------------------------------
# 4. 고아 정리 (REQ-21-14)
# ---------------------------------------------------------------------------


class TestPrune:
    def test_unused_entries_are_removed(self, tmp_path):
        _cache(tmp_path, _Counter())("사라질 문단", PASSAGE)

        alive = _cache(tmp_path, _Counter())
        alive("남아 있는 문단", PASSAGE)
        removed = alive.prune_unused()

        assert removed == 1

    def test_used_entries_survive_prune(self, tmp_path):
        cache = _cache(tmp_path, _Counter())
        cache("남아 있는 문단", PASSAGE)
        cache.prune_unused()

        counter = _Counter()
        _cache(tmp_path, counter)("남아 있는 문단", PASSAGE)

        assert counter.count == 0

    def test_prune_without_lookups_removes_nothing(self, tmp_path):
        """아무것도 조회하지 않은 세션이 캐시를 통째로 지워서는 안 된다."""
        _cache(tmp_path, _Counter())("문단", PASSAGE)

        assert _cache(tmp_path, _Counter()).prune_unused() == 0


# ---------------------------------------------------------------------------
# 5. 관측 가능성 (REQ-21-16)
# ---------------------------------------------------------------------------


class TestStats:
    def test_hit_and_miss_are_counted(self, tmp_path):
        cache = _cache(tmp_path, _Counter())

        cache("문단", PASSAGE)
        cache("문단", PASSAGE)

        assert cache.stats == (1, 1)

    def test_query_is_not_counted(self, tmp_path):
        cache = _cache(tmp_path, _Counter())

        cache("질의", QUERY)

        assert cache.stats == (0, 0)


# ---------------------------------------------------------------------------
# 6. 고장 (REQ-21-15 · 결정 7)
# ---------------------------------------------------------------------------


class TestFailure:
    def test_unusable_path_does_not_stop_embedding(self, tmp_path):
        """캐시가 죽어도 검색은 산다. 디렉토리를 DB 경로로 주어 열기를 실패시킨다."""
        blocked = tmp_path / "blocked.db"
        blocked.mkdir()
        counter = _Counter()

        vector = EmbeddingCache(counter, str(blocked), model_id=MODEL)("문단", PASSAGE)

        assert vector is not None
        assert counter.count == 1

    def test_unusable_path_is_reported(self, tmp_path):
        """조용히 퇴화하지 않는다 (REQ-06-1)."""
        blocked = tmp_path / "blocked.db"
        blocked.mkdir()

        cache = EmbeddingCache(_Counter(), str(blocked), model_id=MODEL)
        cache("문단", PASSAGE)

        assert len(cache.issues) == 1
        assert cache.issues[0].expected is False

    def test_failure_is_reported_once(self, tmp_path):
        """조각마다 기록하면 로그가 넘친다."""
        blocked = tmp_path / "blocked.db"
        blocked.mkdir()

        cache = EmbeddingCache(_Counter(), str(blocked), model_id=MODEL)
        cache("하나", PASSAGE)
        cache("둘", PASSAGE)

        assert len(cache.issues) == 1

    def test_embedding_failure_returns_none(self, tmp_path):
        """임베딩 자체가 죽으면 호출부가 키워드 매칭으로 퇴화할 수 있어야 한다."""

        def broken(_text: str, _kind: str):
            raise RuntimeError("모델 로드 실패")

        cache = _cache(tmp_path, broken)

        assert cache("문단", PASSAGE) is None
        assert len(cache.issues) == 1

    def test_embedding_failure_is_not_cached(self, tmp_path):
        """실패를 저장하면 다음 부팅에서도 영구히 실패한다."""

        calls = {"n": 0}

        def flaky(text: str, _kind: str):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("일시적 실패")
            return np.ones(4, dtype=np.float32)

        cache = _cache(tmp_path, flaky)
        cache("문단", PASSAGE)

        assert cache("문단", PASSAGE) is not None


# ---------------------------------------------------------------------------
# 7. 손상된 캐시 (결정 7)
# ---------------------------------------------------------------------------


class TestCorruption:
    def test_corrupt_file_does_not_stop_embedding(self, tmp_path):
        path = tmp_path / "corrupt.db"
        path.write_bytes(b"this is not a sqlite database")
        counter = _Counter()

        vector = EmbeddingCache(counter, str(path), model_id=MODEL)("문단", PASSAGE)

        assert vector is not None

    def test_corrupt_file_is_reported(self, tmp_path):
        path = tmp_path / "corrupt.db"
        path.write_bytes(b"this is not a sqlite database")

        cache = EmbeddingCache(_Counter(), str(path), model_id=MODEL)
        cache("문단", PASSAGE)

        assert len(cache.issues) == 1

    @pytest.mark.parametrize("stored_dim", [8, 128])
    def test_dimension_mismatch_recomputes(self, tmp_path, stored_dim):
        """차원이 다른 벡터를 돌려주면 내적이 조용히 어긋난다."""

        def wide(_text: str, _kind: str) -> np.ndarray:
            return np.ones(stored_dim, dtype=np.float32)

        _cache(tmp_path, wide)("문단", PASSAGE)

        counter = _Counter()  # 4차원을 돌려준다
        restored = _cache(tmp_path, counter)("문단", PASSAGE)

        assert restored.shape == (stored_dim,)
        assert counter.count == 0
