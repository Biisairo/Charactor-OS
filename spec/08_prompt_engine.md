# 프롬프트 엔진 스펙

## 1. 목적

CharacterOS의 시스템 프롬프트를 **동적으로 조립**한다.
모든 모듈의 데이터를 토큰 예산에 맞게 지능적으로 선택·배치하여 최적의 프롬프트를 생성한다.

## 2. 현재 vs 개선

| 구분 | 현재 (v1) | 개선 (v2) |
|---|---|---|
| 조립 방식 | 모든 컨텍스트를 순서대로 나열 | 관련성 + 토큰 예산 기반 동적 선택 |
| Few-shot | 없음 | 관련 예시 동적 선택 (3~5개) |
| 행동 지침 | 없음 | 상황별/주제별/절대 규칙 포함 |
| 내면 상태 | 없음 | 생각/숨기는 감정/하고픈 말 포함 |
| 관계 정보 | 없음 | 관련 관계만 선택적 포함 |
| 토큰 관리 | 없음 | 섹션별 예산 배분 |
| 지식 선택 | 전체 포함 | 관련성 기반 선택 |

## 3. 프롬프트 구조

시스템 프롬프트는 **8개 섹션**으로 구성된다:

```
┌─────────────────────────────────────────┐
│ 1. Identity          (필수, ~30 tokens) │  ← 이름 + 한줄 소개
│ 2. Persona           (필수, ~200 tokens)│  ← 성격, 말투, 가치관, 배경
│ 3. Behavior          (필수, ~250 tokens)│  ← 행동 지침, 규칙
│ 4. Inner World       (필수, ~80 tokens) │  ← 내면 상태
│ 5. Emotion           (필수, ~50 tokens) │  ← 현재 감정 상태
│ 6. Few-shot          (선택, ~200 tokens)│  ← 예시 대화 (동적 선택)
│ 7. Knowledge         (선택, ~300 tokens)│  ← 세계관/지식 (관련성 기반)
│ 8. Relationships     (선택, ~100 tokens)│  ← 관계 정보 (필요 시)
│ 9. Memory            (선택, ~200 tokens)│  ← 관련 기억 (검색 기반)
│ 10. History          (필수, ~200 tokens)│  ← 최근 대화
│ 11. Response Guide   (필수, ~100 tokens)│  ← 응답 가이드라인
└─────────────────────────────────────────┘
Total target: ~1,500 tokens (약 6,000자)
```

### 우선순위 분류

| 분류 | 섹션 | 포함 조건 |
|---|---|---|
| **Always** | Identity, Persona, Behavior, Inner World, Emotion, Response Guide | 항상 포함 |
| **Context** | Few-shot, Knowledge, Relationships, Memory | 관련성 + 예산 기반 선택 |
| **Always** | History | 항상 포함 (최근 N개) |

## 4. 동적 조립 알고리즘

```python
class PromptEngine:
    MAX_SYSTEM_TOKENS = 2000  # 시스템 프롬프트 상한

    def assemble(self, user_input: str, context: ContextBundle, modules: dict) -> str:
        """
        Args:
            user_input: 사용자 입력
            context: 3-stage의 ContextBundle
            modules: {
                "persona": PersonaModule,
                "emotion": EmotionModule,
                "memory": MemoryModule,
                "knowledge": KnowledgeModule,
                "history": HistoryModule,
                "fewshot": FewShotModule,
            }
        Returns:
            완성된 시스템 프롬프트 문자열
        """
```

### 조립 순서

```
1. Always 섹션 생성 (Identity ~ Inner World + Response Guide)
   → 고정 비용 계산

2. 남은 예산 = MAX_SYSTEM_TOKENS - always 비용

3. Context 섹션 배분:
   - fewshot_budget = remaining × 0.25
   - knowledge_budget = remaining × 0.35
   - relationship_budget = remaining × 0.10
   - memory_budget = remaining × 0.20
   - history_budget = remaining × 0.10

4. 각 섹션을 예산 내에서 생성:
   - Few-shot: search(query, emotions, token_budget=fewshot_budget)
   - Knowledge: search_relevant(query, token_budget=knowledge_budget)
   - Relationships: 쿼리에 관련 인물이 포함되면 포함
   - Memory: to_prompt(query, token_budget=memory_budget)
   - History: to_prompt(n=calculated_from_budget)

5. 조립 + 응답 가이드라인 추가
```

### 토큰 예산 배분 시각화

```
┌──────────────────────────────────────────────┐
│            MAX_SYSTEM_TOKENS (2000)           │
├──────────────────────────────────────────────┤
│ Always │ Identity   │  30 │ ██                │
│        │ Persona    │ 200 │ ██████████        │
│        │ Behavior   │ 250 │ ████████████      │
│        │ InnerWorld │  80 │ ████              │
│        │ Emotion    │  50 │ ██                │
│        │ RespGuide  │ 100 │ █████             │
│        │ History    │ 200 │ ██████████        │
│        │            │     │                   │
│        │ subtotal   │ 910 │                   │
├────────┼────────────┼─────┼───────────────────┤
│ Context│ remaining  │1090 │                   │
│        │ Few-shot   │ 273 │ ████████████      │
│        │ Knowledge  │ 382 │ █████████████████ │
│        │ Relation   │ 109 │ █████             │
│        │ Memory     │ 218 │ ██████████        │
│        │ History*   │ 108 │ █████             │
└────────┴────────────┴─────┴───────────────────┘
* History는 Always이지만 길이 가변 → Context 예산에서 조절
```

## 5. PromptEngine 클래스

### 인터페이스

```python
class PromptEngine:
    def __init__(self, max_tokens: int = 2000):
        """최대 토큰 수를 설정한다."""

    def assemble_system_prompt(
        self,
        user_input: str,
        persona: PersonaModule,
        emotion: EmotionModule,
        memory: MemoryModule,
        knowledge: KnowledgeModule,
        history: HistoryModule,
        fewshot: FewShotModule,
    ) -> str:
        """모든 모듈의 데이터를 조립하여 시스템 프롬프트를 생성한다."""

    def _build_identity(self, persona: PersonaModule) -> str:
        """Identity 섹션 생성."""

    def _build_persona(self, persona: PersonaModule) -> str:
        """Persona 섹션 생성 (성격, 말투, 가치관, 배경)."""

    def _build_behavior(self, persona: PersonaModule) -> str:
        """Behavior 섹션 생성 (행동 지침, 규칙)."""

    def _build_inner_world(self, persona: PersonaModule) -> str:
        """Inner World 섹션 생성."""

    def _build_emotion(self, emotion: EmotionModule) -> str:
        """Emotion 섹션 생성."""

    def _build_fewshot(
        self,
        query: str,
        emotions: dict,
        fewshot: FewShotModule,
        persona: PersonaModule,
        token_budget: int,
    ) -> str:
        """Few-shot 섹션 생성 (내장+외부 예시 통합)."""

    def _build_knowledge(self, query: str, knowledge: KnowledgeModule, token_budget: int) -> str:
        """Knowledge 섹션 생성 (관련성 기반 선택)."""

    def _build_relationships(
        self, query: str, persona: PersonaModule, knowledge: KnowledgeModule, token_budget: int
    ) -> str:
        """Relationships 섹션 생성."""

    def _build_memory(self, query: str, memory: MemoryModule, token_budget: int) -> str:
        """Memory 섹션 생성."""

    def _build_history(self, history: HistoryModule, token_budget: int) -> str:
        """History 섹션 생성."""

    def _build_response_guide(self) -> str:
        """Response Guide 섹션 생성."""

    def _estimate_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 추정 (한글 1자 ≈ 1.5 tokens)."""
```

## 6. Response Guide (응답 가이드라인)

```
[응답 규칙]
- 위 정보를 바탕으로 자연스럽고 일관된 응답을 생성하세요.
- 캐릭터의 성격과 말투를 유지하세요.
- 감정 상태에 맞게 톤을 조절하세요.
- 행동 지침을 반드시 따르세요.
- 기억된 정보를 활용하여 개인화된 응답을 하세요.
- 진짜 사람과의 대화 같아야 합니다. 과한 연기는 금지.
- 설정은 절대적입니다. 설정과 모순되는 응답을 하면 안 됩니다.
- 내면 상태(생각/숨기는 감정)는 응답에 자연스럽게 녹여내세요.
- 예시 대화의 패턴을 참고하되, 그대로 복사하지 마세요.
```

## 7. 쿼리 분석 (선택적 최적화)

복잡한 입력의 경우, 관련성 판단을 위한 사전 분석:

```python
def _analyze_query(self, user_input: str) -> dict:
    """사용자 입력에서 키워드/의도를 추출하여 각 섹션의 포함 여부를 판단."""
    return {
        "keywords": [...],  # 핵심 키워드
        "mentions_character": [...],  # 언급된 캐릭터 이름
        "scenario_type": "...",  # 인사/위로/갈등/유머/일상
        "needs_knowledge": bool,  # 세계관 지식 필요한지
        "needs_relationships": bool,  # 관계 정보 필요한지
    }
```

이 분석은 LLM 호출 없이 **규칙 기반**으로 수행 (비용 최소화):
- 키워드 매칭으로 시나리오 태그 결정
- 캐릭터 이름 감지로 관계 정보 포함 여부 결정
- 세계관 관련 키워드(시대, 장소, 사건) 감지로 지식 포함 여부 결정

## 8. 검증 기준

| 검증 | 기준 |
|---|---|
| Always 포함 | Identity, Persona, Behavior, InnerWorld, Emotion, History, RespGuide 항상 포함 |
| 토큰 예산 | 총 토큰이 MAX_SYSTEM_TOKENS를 초과하지 않음 |
| Few-shot 선택 | 관련 예시가 동적으로 선택됨 |
| Knowledge 선택 | 관련 지식만 포함됨 |
| 관계 선택 | 캐릭터 이름 언급 시 관계 정보 포함 |
| 빈 모듈 | 모듈 데이터가 없으면 해당 섹션 생략 |
| 조립 순서 | 섹션이 지정된 순서대로 배치됨 |
