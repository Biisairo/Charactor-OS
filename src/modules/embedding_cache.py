"""임베딩 영속 캐시 (SPEC-12 REQ-21-10 ~ 21-16).

같은 문장을 부팅마다 다시 임베딩하던 자리다. Knowledge 는 `load_all()`마다
조각 전체를 재계산했고(P-4), FewShot 은 검색 한 번마다 예시마다 계산했다(P-5).

캐시 키는 **콘텐츠 해시**다 — 파일 mtime 도 파일명도 키에 들어가지 않는다.
그래서 파일을 옮기거나 이름을 바꿔도 살아남고, 문서의 한 조각만 고치면 그
조각만 다시 계산된다 (SPEC-12 결정 2).

모델 식별자가 키에 들어가므로 **모델을 바꾸면 무효화 절차 없이 미스가 된다**
(결정 1).

이 모듈에는 LLM 호출이 없다.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.embedding import PASSAGE, EmbeddingKind
from src.modules.asset_issue import AssetLoadIssue

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key     TEXT PRIMARY KEY,
    model   TEXT NOT NULL,
    dim     INTEGER NOT NULL,
    vector  BLOB NOT NULL,
    used_at REAL NOT NULL
)
"""


class EmbeddingCache:
    """임베딩 함수를 감싸 결과를 영속화한다.

    호출부는 이것을 `embedding_fn` 자리에 그대로 꽂는다 — 캐시가 있는지
    모듈이 알 필요가 없다.

    **질의는 캐시하지 않는다.** 질의는 매번 다르므로 저장하면 캐시가 무한히
    자라기만 한다. 저장 대상은 자산 쪽(`passage`)뿐이다.
    """

    def __init__(self, embedding_fn: Callable[[str, str], np.ndarray], db_path: str, model_id: str):
        self._embedding_fn = embedding_fn
        self._db_path = Path(db_path)
        self._model_id = model_id
        self._conn: sqlite3.Connection | None = None
        self._opened = False
        self._seen: set[str] = set()
        self._hits = 0
        self._misses = 0
        self._issues: list[AssetLoadIssue] = []
        self._cache_failed = False
        self._embedding_failed = False

    @property
    def stats(self) -> tuple[int, int]:
        """(히트, 미스). 캐시가 실제로 듣고 있는지 드러낸다 (REQ-21-16)."""
        return self._hits, self._misses

    @property
    def issues(self) -> list[AssetLoadIssue]:
        return list(self._issues)

    def __call__(self, text: str, kind: EmbeddingKind) -> np.ndarray | None:
        if kind != PASSAGE:
            return self._compute(text, kind)

        key = self._key(text, kind)
        self._seen.add(key)

        cached = self._read(key)
        if cached is not None:
            self._hits += 1
            return cached

        self._misses += 1
        vector = self._compute(text, kind)
        if vector is not None:
            # 실패를 저장하면 다음 부팅에서도 영구히 실패한다. 성공만 남긴다.
            self._write(key, vector)
        return vector

    def prune_unused(self) -> int:
        """이번 세션에 조회되지 않은 항목을 지운다 (REQ-21-14).

        정리하지 않으면 자산을 고칠수록 캐시가 자란다. 아무것도 조회하지 않은
        세션은 아무것도 지우지 않는다 — 캐시를 통째로 날려서는 안 된다.
        """
        if not self._seen:
            return 0

        conn = self._connect()
        if conn is None:
            return 0

        placeholders = ",".join("?" * len(self._seen))
        cursor = conn.execute(
            f"DELETE FROM embeddings WHERE key NOT IN ({placeholders})", tuple(self._seen)
        )
        conn.commit()
        return cursor.rowcount

    # ─── 내부 ───

    def _key(self, text: str, kind: str) -> str:
        """모델·용도·텍스트의 해시. 셋 중 하나라도 다르면 다른 항목이다."""
        digest = hashlib.sha256()
        digest.update(self._model_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(kind.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def _compute(self, text: str, kind: str) -> np.ndarray | None:
        """임베딩. 실패는 한 번만 기록하고 `None`으로 알린다 — 호출부가
        키워드 매칭으로 퇴화할 수 있어야 한다."""
        try:
            return self._embedding_fn(text, kind)
        except Exception as e:
            if not self._embedding_failed:
                self._embedding_failed = True
                self._issues.append(
                    AssetLoadIssue(
                        filename="(임베딩)",
                        reason=f"{type(e).__name__}: {e} — 키워드 매칭으로 퇴화",
                        expected=False,
                    )
                )
            return None

    def _connect(self) -> sqlite3.Connection | None:
        """캐시를 연다. 열 수 없으면 캐시 없이 계속한다 (결정 7)."""
        if self._opened:
            return self._conn

        self._opened = True
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            conn.execute(_SCHEMA)
            conn.commit()
        except Exception as e:
            self._report_cache_failure(e)
            return None

        self._conn = conn
        return conn

    def _read(self, key: str) -> np.ndarray | None:
        conn = self._connect()
        if conn is None:
            return None

        try:
            row = conn.execute(
                "SELECT dim, vector FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
        except Exception as e:
            self._report_cache_failure(e)
            return None

        if row is None:
            return None

        vector = np.frombuffer(row[1], dtype=np.float32)
        # 차원이 어긋난 항목은 없는 것으로 본다. 돌려주면 내적이 조용히 틀린다.
        return vector if vector.shape == (row[0],) else None

    def _write(self, key: str, vector: np.ndarray) -> None:
        conn = self._connect()
        if conn is None:
            return

        stored = np.asarray(vector, dtype=np.float32)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (key, model, dim, vector, used_at)"
                " VALUES (?, ?, ?, ?, strftime('%s', 'now'))",
                (key, self._model_id, int(stored.shape[0]), stored.tobytes()),
            )
            conn.commit()
        except Exception as e:
            self._report_cache_failure(e)

    def _report_cache_failure(self, error: Exception) -> None:
        """조각마다 기록하면 로그가 넘친다. 처음 1회만 남긴다 (REQ-06-1)."""
        if self._cache_failed:
            return
        self._cache_failed = True
        self._conn = None
        self._issues.append(
            AssetLoadIssue(
                filename=f"(임베딩 캐시) {self._db_path}",
                reason=f"{type(error).__name__}: {error} — 캐시 없이 계산으로 계속한다",
                expected=False,
            )
        )
