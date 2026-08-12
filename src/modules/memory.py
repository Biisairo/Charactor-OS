import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ANSI 색상 코드
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"


@dataclass
class MemoryEntry:
    id: str
    content: str
    embedding: np.ndarray
    weight: float = 1.0
    emotion_tags: dict[str, float] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: float = 0.0
    created_at: float = 0.0
    metadata: dict = field(default_factory=dict)


MIN_RELEVANCE_SCORE = 0.3

# 이보다 가까우면 "같은 이야기일 수 있다"고 보고 LLM에 관계를 묻는다.
# 미만이면 묻지 않고 새 기억으로 넣는다 — 호출을 아끼는 지점이다.
SIMILAR_MEMORY_THRESHOLD = 0.7


class MemoryModule:
    """대화에서 핵심 정보를 추출하여 기억으로 저장한다."""

    def __init__(
        self,
        db_path: str,
        embedding_fn,
        debug: bool = False,
        debug_output: Callable[[str], None] | None = None,
    ):
        self._db_path = Path(db_path)
        self._embedding_fn = embedding_fn
        self._memories: dict[str, MemoryEntry] = {}
        self._debug = debug
        self._debug_output = debug_output or (lambda msg: None)

    def _log_debug(self, message: str, data=None) -> None:
        if not self._debug:
            return
        prefix = f"{Colors.GREEN}{Colors.BOLD}[Memory]{Colors.RESET}"
        self._debug_output(f"{prefix} {message}")
        if data is not None:
            if isinstance(data, dict):
                self._debug_output(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                self._debug_output(str(data))

    def load(self) -> None:
        """SQLite에서 기억을 로드한다."""
        self._log_debug(f"load() 호출 <- {self._db_path}")
        if not self._db_path.exists():
            self._log_debug("파일 없음, 로드 스킵")
            return

        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM memories").fetchall()
            for row in rows:
                entry = MemoryEntry(
                    id=row["id"],
                    content=row["content"],
                    embedding=np.frombuffer(row["embedding"], dtype=np.float32),
                    weight=row["weight"],
                    emotion_tags=json.loads(row["emotion_tags"]),
                    access_count=row["access_count"],
                    last_accessed=row["last_accessed"],
                    created_at=row["created_at"],
                    metadata=json.loads(row["metadata"]),
                )
                self._memories[entry.id] = entry
            self._log_debug(f"로드 완료: {len(self._memories)}개 기억")
        finally:
            conn.close()

    def save(self) -> None:
        """기억을 SQLite에 저장한다."""
        self._log_debug(f"save() 호출 -> {self._db_path}")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    weight REAL DEFAULT 1.0,
                    emotion_tags TEXT DEFAULT '{}',
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL,
                    created_at REAL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            for entry in self._memories.values():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memories
                    (id, content, embedding, weight, emotion_tags, access_count, last_accessed, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.id,
                        entry.content,
                        entry.embedding.tobytes(),
                        entry.weight,
                        json.dumps(entry.emotion_tags, ensure_ascii=False),
                        entry.access_count,
                        entry.last_accessed,
                        entry.created_at,
                        json.dumps(entry.metadata, ensure_ascii=False),
                    ),
                )
            conn.commit()
            self._log_debug(f"저장 완료: {len(self._memories)}개 기억")
        finally:
            conn.close()

    def snapshot_count(self) -> int:
        """롤백용 현재 기억 개수를 반환한다."""
        return len(self._memories)

    def pop_last_n(self, n: int) -> None:
        """가장 최근에 추가된 n개 기억을 제거한다 (롤백용)."""
        if n <= 0:
            return
        keys = list(self._memories.keys())
        for key in keys[-n:]:
            del self._memories[key]

    def _retention(self, created_at: float) -> float:
        """망각 곡선: (1 + t_days / a) ^ (-b)"""
        t_days = (time.time() - created_at) / 86400
        a, b = 30, 0.5
        return (1 + t_days / a) ** (-b)

    def _effective_weight(self, entry: MemoryEntry) -> float:
        """감정 팩터와 retention을 적용한 유효 가중치 계산."""
        emotion_factor = 1.0
        if entry.emotion_tags:
            emotion_factor = sum(entry.emotion_tags.values()) / len(entry.emotion_tags)
        retention = self._retention(entry.created_at)
        return entry.weight * emotion_factor * retention

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """가중 유사도 기반으로 관련 기억을 검색한다."""
        self._log_debug(f"search() 호출: query='{query}', top_k={top_k}")

        if not self._memories:
            self._log_debug("기억 없음, 빈 결과 반환")
            return []

        query_vec = self._embedding_fn(query)
        scores: list[tuple[str, float]] = []

        for entry in self._memories.values():
            ew = self._effective_weight(entry)
            score = float(np.dot(query_vec, entry.embedding * ew))
            scores.append((entry.id, score))

        scores.sort(key=lambda x: -x[1])
        results = []
        for entry_id, score in scores:
            if score < MIN_RELEVANCE_SCORE:
                break  # 정렬 상태이므로 이후 항목도 모두 미달
            entry = self._memories[entry_id]
            entry.access_count += 1
            entry.last_accessed = time.time()
            results.append(
                {
                    "id": entry.id,
                    "content": entry.content,
                    "score": score,
                    "weight": entry.weight,
                }
            )
            if len(results) >= top_k:
                break

        self._log_debug(f"검색 결과: {len(results)}개")
        for r in results:
            self._log_debug(
                f"  - {r['content']} (score: {r['score']:.4f}, weight: {r['weight']:.2f})"
            )

        return results

    def to_prompt(self, query: str, top_k: int = 5, token_budget: int = 0) -> str:
        """검색된 기억을 프롬프트 문자열로 변환한다.

        Args:
            query: 검색 쿼리
            top_k: 최대 결과 수
            token_budget: 토큰 예산 (0이면 제한 없음)
        """
        results = self.search(query, top_k)
        if not results:
            return "[관련 기억]\n관련 기억 없음"

        lines = ["[관련 기억]"]
        used_tokens = self._estimate_tokens(lines[0])

        for r in results:
            line = f"- {r['content']} (가중치: {r['weight']:.1f})"
            line_tokens = self._estimate_tokens(line)
            if token_budget > 0 and used_tokens + line_tokens > token_budget:
                break
            lines.append(line)
            used_tokens += line_tokens

        return "\n".join(lines)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """토큰 수 추정."""
        korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
        other_chars = len(text) - korean_chars
        return int(korean_chars * 1.5 + other_chars * 0.3)

    def update(
        self,
        user_input: str,
        character_response: str,
        emotions: dict[str, float],
        extractor,
        classifier,
        history_context: str = "",
    ) -> None:
        """대화에서 뽑은 사실을 기억에 반영한다.

        LLM 상호작용은 `extractor`·`classifier`가 맡는다. 이 메서드는 **무엇을
        저장·병합·갱신할지**만 정하므로, 더블 없이 규칙만 바꿔 끼워 검증할 수 있다.

        Args:
            extractor: `extract(user_input, response, history) -> list[MemoryCandidate]`
            classifier: `classify(existing_content, content) -> Classification`
        """
        self._log_debug("")
        self._log_debug("update() 호출")

        candidates = extractor.extract(user_input, character_response, history_context)
        self._log_debug(f"추출된 기억: {len(candidates)}개")

        now = time.time()
        for i, candidate in enumerate(candidates):
            self._log_debug(
                f"기억 [{i + 1}/{len(candidates)}] 처리 중: "
                f"'{candidate.content}' (importance: {candidate.importance})"
            )
            self._absorb(candidate, emotions, classifier, now)

        self._log_debug(f"총 기억 개수: {len(self._memories)}")

    def _absorb(self, candidate, emotions: dict[str, float], classifier, now: float) -> None:
        """후보 하나를 기존 기억에 흡수하거나 새로 추가한다."""
        # 분석 층도 빈 내용을 거르지만, 빈 기억을 만들지 않는 것은 도메인의 불변식이다.
        # 어느 경로로 들어오든 지켜야 한다.
        if not candidate.content:
            return

        embedding = self._embedding_fn(candidate.content)
        nearest, score = self._nearest(embedding)

        # 충분히 가까운 기억이 없으면 판정을 물을 이유가 없다. 호출을 아낀다.
        if nearest is None or score < SIMILAR_MEMORY_THRESHOLD:
            self._log_debug(f"유사도 {score:.4f} < {SIMILAR_MEMORY_THRESHOLD} -> 새 기억 추가")
            self._insert(candidate, embedding, emotions, now)
            return

        classification = classifier.classify(nearest.content, candidate.content)
        self._log_debug(f"분류 결과: {classification}")

        if classification == "IDENTICAL":
            # 같은 정보다. 내용은 그대로 두고 참조 기록만 남긴다.
            nearest.last_accessed = now
            nearest.access_count += 1
        elif classification == "SIMILAR":
            # 관련 있지만 더 새로운 정보다. 내용을 갱신하되 중요도는 낮추지 않는다.
            self._log_debug(f"  '{nearest.content}' -> '{candidate.content}'")
            nearest.content = candidate.content
            nearest.weight = max(nearest.weight, candidate.importance)
            nearest.last_accessed = now
        else:
            self._insert(candidate, embedding, emotions, now)

    def _nearest(self, embedding) -> tuple[MemoryEntry | None, float]:
        """임베딩과 가장 가까운 기억과 그 유사도를 반환한다."""
        best, best_score = None, -1.0
        for entry in self._memories.values():
            score = float(np.dot(embedding, entry.embedding))
            if score > best_score:
                best, best_score = entry, score
        return best, best_score

    def _insert(self, candidate, embedding, emotions: dict[str, float], now: float) -> None:
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=candidate.content,
            embedding=embedding,
            weight=candidate.importance,
            emotion_tags=dict(emotions),
            access_count=0,
            last_accessed=now,
            created_at=now,
        )
        self._memories[entry.id] = entry
        self._log_debug(f"  새 기억 추가: id={entry.id}, content='{candidate.content}'")
