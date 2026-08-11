# Few-shot 예시 시스템 스펙

## 1. 목적

캐릭터의 응답 패턴을 **예시 대화**로 LLM에 학습시킨다.
"이런 말에 이렇게 대답한다"는 패턴을 few-shot으로 제공하여 캐릭터 일관성을 높인다.

## 2. Few-shot vs Persona examples 구분

| 구분 | Persona examples (내장) | Few-shot 시스템 (외부) |
|---|---|---|
| 위치 | persona.yaml 내부 | `examples/` 디렉토리 |
| 용도 | 핵심 성격 보여주기 | 다양한 상황 커버 |
| 관리 | 캐릭터 작성자가 관리 | 시스템이 관리 (추가/수정 가능) |
| 검색 | 항상 포함 (최대 3개) | 관련성 기반 동적 선택 |

## 3. Few-shot 데이터 구조

### 파일 구조

```
examples/
├── greeting.yaml           # 인사 관련
├── comfort.yaml            # 위로 관련
├── conflict.yaml           # 갈등 관련
├── humor.yaml              # 유머 관련
└── daily.yaml              # 일상 대화
```

### YAML 형식

```yaml
# examples/comfort.yaml
tag: "위로"                    # 시나리오 태그
examples:
  - user: "오늘 시험 망했어..."
    character: "아... 속상하겠다. 어디가 어려웠어?"
    emotion_state:              # 이 예시가 적용되는 감정 상태 (선택)
      - "슬픔"
      - "실망"

  - user: "친구랑 싸웠어"
    character: "헐 왜? 무슨 일이야?"
    emotion_state:
      - "슬픔"
      - "분노"
```

### 전체 구조

```python
@dataclass
class FewShotExample:
    user: str  # 사용자 발화
    character: str  # 캐릭터 응답
    emotion_state: list[str]  # 적용 감정 상태 (선택)


@dataclass
class FewShotGroup:
    tag: str  # 시나리오 태그
    examples: list[FewShotExample]
```

## 4. 검색 알고리즘

Few-shot 예시는 사용자 입력과의 **관련성**에 따라 동적으로 선택된다.

### 검색 점수

```
score = tag_match × 0.4 + semantic_similarity × 0.4 + emotion_match × 0.2
```

| 요소 | 가중치 | 설명 |
|---|---|---|
| tag_match | 0.4 | 시나리오 태그와 사용자 입력의 키워드 매칭 |
| semantic_similarity | 0.4 | 임베딩 유사도 (MemoryModule과 동일한 모델) |
| emotion_match | 0.2 | 현재 감정 상태와 예시의 감정 상태 매칭 |

### 선택 로직

1. 사용자 입력 → 키워드 추출 + 임베딩
2. 각 FewShotGroup의 tag와 키워드 매칭
3. 각 FewShotGroup의 examples와 임베딩 유사도 계산
4. 현재 감정 상태와 emotion_state 매칭
5. 점수 합산 → 상위 N개 선택 (기본 3개)
6. token_budget 내에서 포함

### 키워드 매칭 규칙

| 태그 | 트리거 키워드 |
|---|---|
| 인사 | 안녕, 하이, 헬로, 반가워 |
| 위로 | 힘들어, 슬퍼, 우울, 망했어, 실패 |
| 갈등 | 싸웠어, 화나, 짜증, 미워 |
| 유머 | 웃겨, 농담, ㅋㅋ, 재밌어 |
| 일상 | 뭐해, 밥, 오늘, 취미 |

## 5. FewShotModule 클래스

### 인터페이스

```python
class FewShotModule:
    def __init__(self, examples_dir: str, embedding_fn: Callable):
        """예시 디렉토리와 임베딩 함수를 설정한다."""

    def load_all(self) -> None:
        """examples/ 디렉토리의 모든 YAML을 로드한다."""

    def search(
        self, query: str, emotions: dict[str, float], top_k: int = 3
    ) -> list[FewShotExample]:
        """관련성 기반으로 few-shot 예시를 검색한다."""

    def to_prompt(
        self, query: str, emotions: dict[str, float], top_k: int = 3, token_budget: int = 300
    ) -> str:
        """검색된 예시를 프롬프트 문자열로 변환한다."""

    def add_example(
        self, tag: str, user: str, character: str, emotion_state: list[str] | None = None
    ) -> None:
        """새 예시를 추가한다 (런타임)."""

    def get_all_tags(self) -> list[str]:
        """모든 태그 목록을 반환한다."""
```

### to_prompt() 출력 형식

```
[예시 대화]
상황: 위로
사용자: 오늘 시험 망했어...
캐릭터: 아... 속상하겠다. 어디가 어려웠어?

상황: 인사
사용자: 안녕!
캐릭터: 안녕! 반가워 😊
```

예시가 없으면:
```
[예시 대화]
예시 없음
```

## 6. 영속화

- 정적 데이터: `examples/` YAML 파일 (사용자 관리)
- 동적 추가: `add_example()`로 런타임에 추가 가능
  - 저장: `examples/custom.yaml`에 누적
  - 프로그램 재시작 후 유지

## 7. Persona 내장 예시와의 통합

PersonaModule의 `examples` 필드와 FewShotModule의 예시를 **합쳐서** 검색한다:

```python
# PromptEngine에서의 사용
persona_examples = persona.get_examples()  # 내장 예시
fewshot_examples = fewshot.search(query, emotions)  # 외부 예시
all_examples = persona_examples + fewshot_examples
# 중복 제거, 점수 정렬, token_budget 내 선택
```

## 8. 검증 기준

| 검증 | 기준 |
|---|---|
| YAML 로드 | examples/ 디렉토리의 모든 YAML 파싱 |
| 태그 검색 | 키워드 매칭 → 해당 태그의 예시 반환 |
| 임베딩 검색 | 유사도 기반 관련 예시 반환 |
| 감정 매칭 | 현재 감정과 일치하는 예시 우선 |
| 프롬프트 변환 | 예시가 문자열로 정확히 변환됨 |
| 빈 디렉토리 | "예시 없음" 반환 |
| 동적 추가 | add_example → search에서 검색 가능 |
| 토큰 예산 | token_budget 초과 시 잘림 |
