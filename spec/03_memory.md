# Memory 모듈 스펙

## 1. 목적

대화에서 핵심 정보를 추출하여 기억으로 저장한다.
기존 weighted_memory(04_rag_advanced)의 방식을 기반으로 캐릭터 전용으로 구현한다.

## 2. Memory vs Knowledge 구분

| 구분 | Memory (이 모듈) | Knowledge |
|---|---|---|
| 내용 | 대화에서 발생한 정보 | 사용자가 주입한 지식 |
| 관리자 | 에이전트 | 사용자 |
| 예시 | "사용자가 서울 산다", "어제 영화 봤다" | "서울 인구 960만", "AI 역사" |
| 수정 | 에이전트가 대화로 업데이트 | 사용자만 수정 |

## 3. 기존 weighted_memory 분석

기존 구현(`04_rag_advanced/weighted_memory/weighted_memory.py`)의 핵심:

- **FAISS 안 씀** — numpy dot product로 벡터 유사도 계산
- **SQLite**로 영속화
- **메모리 내 dict** (`self.vectors`)로 관리
- **Power-law retention** — `(1 + t/a)^(-b)` 망각 곡선
- **가중치**: importance, emotion, 신뢰도 기반
- **충돌 판정**: 유사 기억 발견 시 LLM으로 IDENTICAL/SIMILAR/DIFFERENT 판단

## 4. 기억 데이터 구조

기존 `WeightedVec` 기반:

```python
@dataclass
class MemoryEntry:
    id: str  # 고유 ID
    content: str  # 기억 내용
    embedding: np.ndarray  # 벡터 (L2 정규화)
    weight: float  # 기본 가중치 (0.1 ~ 3.0)
    emotion_tags: dict[str, float]  # 저장 시점의 감정 태그
    access_count: int  # 접근 횟수
    last_accessed: float  # 마지막 접근 시각
    created_at: float  # 생성 시각
    metadata: dict  # 추가 메타데이터
```

## 5. 가중치 검색 점수

기존 weighted_memory와 동일한 방식:

```
effective_weight = base_weight × emotion_factor × retention

score = dot(query_vec, embedding × effective_weight)
```

### retention (망각 곡선)

```
retention = (1 + t_days / a) ^ (-b)

a = 30 (반감기 조절)
b = 0.5 (감소 속도)
```

### emotion_factor

저장 시점 감정 태그의 평균값. 감정이 강할수록 기억이 선명.

## 6. MemoryModule 클래스

### 인터페이스

```python
class MemoryModule:
    def __init__(self, db_path: str, embedding_fn: Callable):
        """DB 경로와 임베딩 함수를 설정한다."""

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """가중 유사도 기반으로 관련 기억을 검색한다."""

    def update(
        self, user_input: str, character_response: str, emotions: dict[str, float], client: Client
    ) -> None:
        """대화에서 핵심 정보를 추출하여 기억에 저장한다. (별도 LLM 호출)"""

    def to_prompt(self, query: str, top_k: int = 5) -> str:
        """검색된 기억을 프롬프트 문자열로 변환한다."""

    def save(self) -> None:
        """기억을 SQLite에 저장한다."""

    def load(self) -> None:
        """SQLite에서 기억을 로드한다."""
```

### update() LLM 프롬프트

```
다음 대화에서 **사용자에 대한 구체적인 사실**만 추출하세요.

{이전 대화 맥락 - 흐름과 맥락을 보기 위함}

사용자: {user_input}
캐릭터: {character_response}

다음 JSON 형식으로 반환하세요:
{
    "memories": [
        {
            "content": "기억할 내용",
            "importance": 0.0~1.0
        }
    ]
}

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
- 추출할 정보가 없으면 빈 배열 반환
```

**참고**: 이전 대화 맥락은 사용자의 패턴을 이해하기 위함이며, 실제 기억 추출은 이번 대화를 기반으로 수행한다. 감정 상태는 Emotion 모듈이 별도 관리하므로 기억하지 않는다.

### to_prompt() 출력 형식

```
[관련 기억]
- 사용자는 서울 산다 (가중치: 1.8)
- 어제 영화를 봤다 (가중치: 1.2)
```

기억이 없으면:
```
[관련 기억]
관련 기억 없음
```

## 7. 검색 알고리즘

기존 weighted_memory의 `_enhanced_embed` + query_enhancement의 쿼리 강화 결합:

```
1. 원본 쿼리 → _enhanced_embed (원문 + 토큰 임베딩 평균)
2. 키워드 추출 (LLM) → 핵심 단어 추출
3. Query Expansion (필요 시) → 동의어/관련어 확장
4. 각 쿼리로 numpy dot product 검색
   score = dot(query_vec, embedding × effective_weight)
5. 결과 합산 (중복 제거, 높은 점수 유지)
6. 가중치 점수 기준 정렬 → 상위 top_k 반환
```

### 쿼리 강화 상세

| 레벨 | 기법 | 출처 | 비용 | 설명 |
|---|---|---|---|---|
| 1 | `_enhanced_embed` | weighted_memory | 낮음 | 원문 + 토큰 임베딩 평균 (항상) |
| 2 | 키워드 추출 | LLM 호출 | 중간 | 핵심 단어 추출 → 별도 검색 |
| 3 | Query Expansion | query_enhancement | 중간 | 동의어/관련어 확장 (필요 시) |

- 레벨 1은 항상 적용 (저비용)
- 레벨 2는 충돌 감지 또는 정확도 필요 시 적용
- 레벨 3은 충돌 감지 시에만 적용

### 키워드 추출 LLM 프롬프트

```
다음 문장에서 검색에 쓸 핵심 키워드를 추출하세요.

문장: {query}

JSON 배열로 반환하세요: ["키워드1", "키워드2", ...]
```

추출된 키워드 각각으로 검색 → 결과 합산 (중복 제거, 높은 점수 유지)

## 8. 영속화

**SQLite** 사용 (기존 weighted_memory 패턴):

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,        -- numpy array를 bytes로 저장
    weight REAL DEFAULT 1.0,
    emotion_tags TEXT DEFAULT '{}',  -- JSON 문자열
    access_count INTEGER DEFAULT 0,
    last_accessed REAL,
    created_at REAL,
    metadata TEXT DEFAULT '{}'       -- JSON 문자열
);
```

- SQLite 파일: `memory/memories.db`
- 시작 시 SQLite 로드 → 메모리 내 dict 관리

## 9. 충돌 처리

기존 weighted_memory의 충돌 판정 방식 활용:

- 새 기억 추가 시 기존 기억과 유사도 검사
- 유사도 > 임계값이면 LLM으로 판정:
  - **IDENTICAL**: 새 것 저장 안 함, 기존 것 refresh
  - **SIMILAR**: 기존 것에 merge + refresh
  - **DIFFERENT**: 새로 저장

## 10. 검증 기준

| 검증 | 기준 |
|---|---|
| 기억 추출 | 대화 → LLM → MemoryEntry 리스트 생성 |
| 가중치 검색 | 쿼리 → importance/retention/emotion 반영된 결과 |
| 벡터 검색 | numpy dot product로 유사 기억 검색 |
| 영속화 | save → load 후 기억 복원 |
| 감정 태그 | 저장 시점의 감정이 태그로 포함됨 |
| 충돌 처리 | 중복 기억 자동 병합 |
