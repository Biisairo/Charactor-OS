# 모듈 아키텍처 스펙

## 1. 모듈 인터페이스 공통 규칙

모든 모듈은 다음 인터페이스를 따른다:

```python
class BaseModule:
    def to_prompt(self, **kwargs) -> str:
        """이 모듈의 현재 상태를 프롬프트 문자열로 변환한다."""
```

- 정적 모듈 (Persona, Knowledge, Few-shot): `update()` 없음
- 동적 모듈 (Emotion, Memory, History): `update()` 있음 (Stage 3에서 호출)

## 2. 모듈 의존 관계

```
CharacterOS
    │
    ├── PromptEngine             (순수 조립기, 모듈 참조 없음)
    │
    ├── ReActBrain               (Stage 1, ToolRegistry로 모듈 호출)
    │   └── WorkingMemoryModule  (동적, 미해결 질문·가설)
    │
    ├── PersonaModule            (정적, 독립)
    │   └── emotion_triggers → EmotionModule에서 참조
    │
    ├── KnowledgeModule          (정적, 독립)
    │   └── relationships → PromptEngine에서 참조
    │
    ├── FewShotModule            (정적, 독립)
    │   └── embedding_fn → MemoryModule과 동일 모델
    │
    ├── EmotionModule            (동적, 독립)
    │   └── emotion_triggers ← PersonaModule에서 주입
    │
    ├── MemoryModule             (동적, 독립)
    │   └── SentenceTransformer (임베딩)
    │
    └── HistoryModule            (동적, 독립)
```

### 의존 방향

| 관계 | 방향 | 설명 |
|---|---|---|
| Persona → Emotion | 간접 | emotion_triggers를 CharacterOS가 Emotion에 전달 |
| PromptEngine → All | **없음** | 엔진은 ThoughtBundle과 persona만 받는다 (SPEC-09) |
| ReActBrain → All | 직접 | ToolRegistry를 통해 도구로 호출 |
| FewShot ↔ Memory | 공유 | 동일한 embedding_fn 사용 |
| Emotion → Memory | 간접 | 감정 태그를 CharacterOS가 Memory에 전달 |

## 3. CharacterOS 오케스트레이터

```python
class CharacterOS:
    def __init__(self, ...):
        # 정적 모듈
        self.persona = PersonaModule(persona_path)
        self.knowledge = KnowledgeModule(knowledge_dir)
        self.fewshot = FewShotModule(examples_dir, embedding_fn=self._embed)

        # 동적 모듈
        self.emotion = EmotionModule(...)
        self.memory = MemoryModule(...)
        self.history = HistoryModule(...)

        # 프롬프트 엔진
        self.prompt_engine = PromptEngine(max_tokens=2000)

        # LLM 클라이언트
        self.client = CachedClient(Client())

        # 초기화
        self.persona.load()
        self.knowledge.load_all()
        self.fewshot.load_all()
        self.emotion.load()
        self.memory.load()
        self.history.load()

        # 감정 트리거 주입
        self.emotion.set_triggers(self.persona.get_emotion_triggers())

    def chat(self, user_input: str) -> str:
        # Stage 1: 뇌 — 무엇이 필요한지 판단하고 도구로 모은다
        bundle = self._think(user_input)

        # Stage 2: 프롬프트 조립 + 응답 생성 (뇌가 모은 것만 쓴다)
        system_prompt = self.prompt_engine.assemble_system_prompt(self.persona, bundle)
        response = self._generate_response(user_input, bundle)

        # Stage 3: 후처리 (작업기억 포함)
        self._post_process(user_input, response, bundle)

        return response
```

### _think()

```python
def _think(self, user_input: str) -> ThoughtBundle:
    """Stage 1: 뇌가 도구를 골라 근거를 모으고 응답 전략을 세운다."""
    return self.brain.think(user_input)
```

**PromptEngine은 모듈을 참조하지 않는다** (SPEC-09 REQ-RA-30). 예전에는 모듈 참조를
넘겨 엔진이 직접 검색했고, 그 결과 Stage 1과 엔진이 같은 검색을 두 번 돌았다.
이제 검색의 단일 실행 지점은 뇌이며, 번들에 담긴 것만 프롬프트가 된다.

### _generate_response()

```python
def _generate_response(self, system_prompt: str, user_input: str) -> str:
    """Stage 2: 조립된 프롬프트로 응답 생성."""
    result = self.client.call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        use_stream=True,
        mute=True,
    )
    return result.content
```

### _post_process()

```python
def _post_process(self, user_input: str, response: str) -> None:
    """Stage 3: 후처리 (기존과 동일, 감정 트리거 추가)."""
    # 롤백 스냅샷
    emotion_snap = self.emotion.snapshot()
    memory_count = self.memory.snapshot_count()
    history_count = self.history.count()

    history_context = self.history.to_prompt(n=10)

    try:
        # Emotion + History 병렬
        with ThreadPoolExecutor(max_workers=2) as pool:
            emotion_future = pool.submit(
                self.emotion.update,
                user_input,
                response,
                self.client,
                history_context=history_context,
            )
            history_future = pool.submit(self._add_history, user_input, response)
            emotion_future.result()
            history_future.result()

        # Memory (감정 결과 사용)
        self.memory.update(
            user_input,
            response,
            self.emotion.get_state(),
            self.client,
            history_context=history_context,
        )

    except Exception:
        self.emotion.restore(emotion_snap)
        self.memory.pop_last_n(self.memory.snapshot_count() - memory_count)
        self.history.pop_last_n(self.history.count() - history_count)
        raise

    # 영속화
    self.emotion.save()
    self.memory.save()
    self.history.save()
```

## 4. 각 모듈 내부 아키텍처

### PersonaModule

```
YAML
    │
    ▼
load() → dict
    │
    ├── to_system_prompt()         → 기본 프롬프트 (Identity~Persona)
    ├── get_behavior_section()     → 행동 지침 문자열
    ├── get_emotion_triggers()     → 감정 트리거 리스트
    ├── get_examples()             → 내장 few-shot 예시
    ├── get_relationships()        → 관계 설정 리스트
    └── get_inner_world()          → 내면 상태 문자열
```

### KnowledgeModule

```
knowledge/ 디렉토리
    │
    ▼
load_all()
    │
    ├── type: world     → _world dict
    ├── type: character → _characters list
    ├── type: relationships → _relationships list
    ├── type: timeline  → _timeline list
    ├── type: locations → _locations list
    └── type: freeform / 미지정 → _freeform list (기존 호환)
    │
    ├── to_prompt()           → 전체 지식 문자열 (기존 호환)
    ├── get_world()           → 세계관 dict
    ├── get_characters()      → 캐릭터 리스트
    ├── get_relationships()   → 관계 리스트
    ├── get_timeline()        → 타임라인 리스트
    ├── get_locations()       → 장소 리스트
    └── search_relevant()     → 관련 지식만 선택
```

### FewShotModule

```
examples/ 디렉토리
    │
    ▼
load_all()
    │
    ├── YAML 파싱 → FewShotGroup list
    └── 각 example의 태그/감정 인덱싱
    │
    ├── search(query, emotions, top_k)  → 관련 예시 선택
    │   ├── 태그 키워드 매칭 (0.4)
    │   ├── 임베딩 유사도 (0.4)
    │   └── 감정 상태 매칭 (0.2)
    │
    └── to_prompt(query, emotions, top_k, token_budget)
        → 프롬프트 문자열
```

### PromptEngine

```
사용자 입력 + 모듈 참조
    │
    ▼
assemble_system_prompt()
    │
    ├── _build_identity()       → 이름 + 소개
    ├── _build_persona()        → 성격, 말투, 가치관, 배경
    ├── _build_behavior()       → 행동 지침
    ├── _build_inner_world()    → 내면 상태
    ├── _build_emotion()        → 현재 감정
    │
    ├── 남은 예산 계산
    │
    ├── _build_fewshot()        → 예시 대화 (예산 내)
    ├── _build_knowledge()      → 관련 지식 (예산 내)
    ├── _build_relationships()  → 관계 정보 (필요 시)
    ├── _build_memory()         → 관련 기억 (예산 내)
    ├── _build_history()        → 최근 대화 (예산 내)
    │
    ├── _build_response_guide() → 응답 가이드라인
    │
    └── 조립 → 시스템 프롬프트 문자열
```

### EmotionModule

```
                    ┌─────────────────┐
                    │  상태 저장       │
                    │  dict[str, float]│
                    └──────┬──────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
    get_state()      to_prompt()        apply_decay()
                           │
                           ▼
                    update(user, resp, client)
                    │
                    ├── 감정 트리거 매칭 (persona_triggers)
                    │   "아버지" → 분노 0.7
                    │
                    └── LLM 호출 (별도)
                        → 감정 상태 병합
```

## 5. 검증 기준

| 검증 | 기준 |
|---|---|
| PersonaModule | YAML → 변환 후 모든 필드 접근 |
| KnowledgeModule | type별 파싱 → 각 getter 반환 |
| FewShotModule | YAML 로드 → 태그/임베딩 검색 → 프롬프트 변환 |
| PromptEngine | 토큰 예산 내 조립 → Always 섹션 항상 포함 |
| EmotionModule | 트리거 기반 감정 변동 확인 |
| 통합 파이프라인 | 3-stage 전체 동작 → 일관된 캐릭터 응답 |
| 롤백 | Stage 3 실패 시 모든 상태 복원 |
