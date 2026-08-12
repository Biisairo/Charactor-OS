import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.validity import provider_error_reason


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

    def _check_conflict(
        self, content: str, client, prompt_callback: Callable[[str, str], None] | None = None
    ) -> str:
        """기존 기억과 충돌 여부를 확인한다."""
        self._log_debug(f"_check_conflict() 호출: '{content}'")

        if not self._memories:
            self._log_debug("기존 기억 없음 -> DIFFERENT")
            return "DIFFERENT"

        query_vec = self._embedding_fn(content)
        best_id, best_score = None, -1.0

        for entry in self._memories.values():
            score = float(np.dot(query_vec, entry.embedding))
            if score > best_score:
                best_score = score
                best_id = entry.id

        self._log_debug(f"최고 유사도: {best_score:.4f} (id: {best_id})")

        if best_score < 0.7:
            self._log_debug("유사도 < 0.7 -> DIFFERENT")
            return "DIFFERENT"

        existing = self._memories[best_id]
        prompt = f"""다음 두 기억을 비교하세요.

기존 기억: {existing.content}
새 기억: {content}

다음 중 하나로 분류하세요:
- IDENTICAL: 같은 정보
- SIMILAR: 관련 있지만 다른 정보
- DIFFERENT: 완전히 다른 정보

JSON으로 반환: {{"classification": "IDENTICAL|SIMILAR|DIFFERENT"}}"""

        self._log_debug("충돌 판정 LLM 호출")
        self._log_debug(f"기존 기억: {existing.content}")
        self._log_debug(f"새 기억: {content}")

        if prompt_callback:
            prompt_callback("memory_conflict", prompt)

        result = client.call_llm(
            messages=[
                {"role": "system", "content": "기억 분류기. JSON만 반환하세요."},
                {"role": "user", "content": prompt},
            ],
            tools=[],
            use_stream=False,
            mute=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "memory_conflict",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "classification": {
                                "type": "string",
                                "enum": ["SAME", "CONTRADICT", "DIFFERENT"],
                            },
                        },
                        "required": ["classification"],
                    },
                },
            },
        ).content

        self._log_debug(f"LLM 응답: {result}")

        try:
            data = json.loads(result)
            classification = data.get("classification", "DIFFERENT")
            self._log_debug(f"분류 결과: {classification}")
            return classification
        except (json.JSONDecodeError, AttributeError) as e:
            reason = provider_error_reason(result)
            if reason:
                # 파싱 실패로 뭉뚱그리면 프로바이더 장애가 데이터 문제로 보인다.
                self._log_debug(f"프로바이더 거부 — 충돌 판정을 DIFFERENT로 처리: {reason}")
            else:
                self._log_debug(f"JSON 파싱 실패: {e}")
            return "DIFFERENT"

    def update(
        self,
        user_input: str,
        character_response: str,
        emotions: dict[str, float],
        client,
        history_context: str = "",
        prompt_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        """대화에서 핵심 정보를 추출하여 기억에 저장한다. (별도 LLM 호출)

        Args:
            user_input: 사용자 입력
            character_response: 캐릭터 응답
            emotions: 감정 태그
            client: LLM 클라이언트
            history_context: 이전 대화 맥락 (흐름과 맥락을 보기 위함)
            prompt_callback: 프롬프트 로깅 콜백 (module, prompt)
        """
        self._log_debug("")
        self._log_debug("update() 호출")
        self._log_debug(f"사용자 입력: {user_input}")
        self._log_debug(f"캐릭터 응답: {character_response[:50]}...")
        self._log_debug(f"감정 태그: {emotions}")

        prompt = f"""다음 대화에서 **사용자에 대한 구체적인 사실**만 추출하세요.

{history_context}

사용자: {user_input}
캐릭터: {character_response}

다음 JSON 형식으로 반환하세요:
{{
    "memories": [
        {{
            "content": "기억할 내용",
            "importance": 0.0~1.0
        }}
    ]
}}

기억할 수 있는 것 (구체적 사실):
- 이름, 나이, 직업, 거주지
- 좋아하는/싫어하는 것
- 가족, 반려동물
- 특별한 경험, 사건
- 고민, 목표

기억하면 안 되는 것:
- 대화 스타일, 패턴
- 감정 상태 (별도 관리됨)
- 일반적인 관심표현
- 모호한 추론

규칙:
- 이미 알려진 정보와 중복되면 저장하지 않음
- 구체적인 사실만 저장 (추론 X)
- 추출할 정보가 없으면 빈 배열 반환"""

        self._log_debug("기억 추출 LLM 호출")

        if prompt_callback:
            prompt_callback("memory", prompt)

        result = client.call_llm(
            messages=[
                {"role": "system", "content": "기억 추출기. JSON만 반환하세요."},
                {"role": "user", "content": prompt},
            ],
            tools=[],
            use_stream=False,
            mute=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "memory_extraction",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "memories": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "importance": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                        },
                                    },
                                    "required": ["content"],
                                },
                            },
                        },
                        "required": ["memories"],
                    },
                },
            },
        ).content

        self._log_debug(f"LLM 응답: {result}")

        try:
            data = json.loads(result)
            memories = data.get("memories", [])
            self._log_debug(f"추출된 기억: {len(memories)}개")
        except (json.JSONDecodeError, AttributeError) as e:
            reason = provider_error_reason(result)
            if reason:
                # 파싱 실패로 뭉뚱그리면 프로바이더 장애가 데이터 문제로 보인다.
                self._log_debug(f"프로바이더 거부 — 기억 추출을 건너뜀: {reason}")
            else:
                self._log_debug(f"JSON 파싱 실패: {e}")
            return

        now = time.time()
        for i, mem in enumerate(memories):
            content = mem.get("content", "")
            importance = mem.get("importance", 0.5)
            if not content:
                continue

            self._log_debug(
                f"기억 [{i + 1}/{len(memories)}] 처리 중: '{content}' (importance: {importance})"
            )

            conflict = self._check_conflict(content, client, prompt_callback)
            if conflict == "IDENTICAL":
                self._log_debug("IDENTICAL: 기존 기억 갱신")
                for entry in self._memories.values():
                    if float(np.dot(self._embedding_fn(content), entry.embedding)) > 0.7:
                        entry.last_accessed = now
                        entry.access_count += 1
                        break
            elif conflict == "SIMILAR":
                self._log_debug("SIMILAR: 기존 기억 병합")
                for entry in self._memories.values():
                    if float(np.dot(self._embedding_fn(content), entry.embedding)) > 0.7:
                        old_content = entry.content
                        entry.content = content
                        entry.weight = max(entry.weight, importance)
                        entry.last_accessed = now
                        self._log_debug(f"  '{old_content}' -> '{content}'")
                        break
            else:
                self._log_debug("DIFFERENT: 새 기억 추가")
                embedding = self._embedding_fn(content)
                entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    content=content,
                    embedding=embedding,
                    weight=importance,
                    emotion_tags=dict(emotions),
                    access_count=0,
                    last_accessed=now,
                    created_at=now,
                )
                self._memories[entry.id] = entry
                self._log_debug(f"  새 기억 추가: id={entry.id}, content='{content}'")

        self._log_debug(f"총 기억 개수: {len(self._memories)}")
