# Character OS — 전체 아키텍처 스펙

## 1. 개요

Character OS는 하나의 살아있는 캐릭터를 통합하는 시스템이다.
캐릭터는 대화를 통해 감정이 변하고, 기억을 쌓으며, 일관된 성격과 말투로 반응한다.
Persona 스키마 확장, Knowledge 구조화, Few-shot 예시 시스템, 프롬프트 엔진이 포함된다.

## 2. 핵심 설계 원칙

| 원칙 | 설명 |
|---|---|
| **모듈화** | 각 기능은 독립된 모듈 |
| **정적 vs 동적 분리** | Persona, Knowledge, Few-shot은 정적. Emotion, Memory, History는 동적 |
| **에이전트는 정적 파일을 수정하지 않음** | Persona, Knowledge 파일은 절대 에이전트가 수정하지 않음 |
| **프롬프트 엔진이 중심** | 모든 모듈의 데이터를 PromptEngine이 동적으로 조립 |
| **토큰 예산 관리** | 시스템 프롬프트는 토큰 상한 내에서 관련성 기반 선택 |

## 3. 모듈 목록

| 모듈 | 타입 | 파일 | 스키마 | 설명 |
|---|---|---|---|---|
| **Persona** | 정적 | `src/modules/persona.py` | - | 성격, 말투, 행동 지침, 감정 트리거, 내면 상태, 관계, few-shot |
| **Emotion** | 동적 | `src/modules/emotion.py` | - | 감정 상태 추적, 0~1 스케일, 트리거 기반 |
| **Memory** | 동적 | `src/modules/memory.py` | - | 대화에서 핵심 정보 추출, 가중치 기반 관리 |
| **Knowledge** | 정적 | `src/modules/knowledge.py` | - | 세계관, 관계 그래프, 타임라인, 장소, 자유 형식 |
| **History** | 동적 | `src/modules/history.py` | - | 대화 기록 관리, 최근 N개 조회 |
| **FewShot** | 정적 | `src/modules/fewshot.py` | 신규 | 예시 대화 검색, 태그/임베딩/감정 기반 |
| **PromptEngine** | - | `src/prompts/engine.py` | 신규 | 동적 프롬프트 조립, 토큰 예산 관리 |

## 4. 3단계 파이프라인

```
사용자 입력
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: Context Gathering                                  │
│                                                              │
│  사용자 입력 분석 → 각 모듈에서 컨텍스트 수집                    │
│                                                              │
│  수집 대상:                                                    │
│  - persona (항상): 성격, 말투, 행동 지침, 내면 상태             │
│  - emotion (항상): 현재 감정 상태                              │
│  - knowledge (선택): 세계관, 캐릭터, 관계 (관련성 기반)         │
│  - fewshot (선택): 예시 대화 (관련성 + 감정 기반)              │
│  - memory (선택): 관련 기억 (벡터 검색)                        │
│  - history (항상): 최근 N개 대화                               │
│                                                              │
│  산출물: 각 모듈의 raw 데이터                                  │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: Response Generation                                │
│                                                              │
│  PromptEngine이 모든 모듈 데이터를 동적 조립:                   │
│  1. Always 섹션 (Identity, Persona, Behavior, InnerWorld,     │
│     Emotion, ResponseGuide)                                  │
│  2. Context 섹션 예산 배분:                                    │
│     Few-shot 25% + Knowledge 35% + Relations 10%             │
│     + Memory 20% + History 10%                               │
│  3. 토큰 상한 내에서 조립 완료                                  │
│                                                              │
│  완성된 시스템 프롬프트 + 사용자 입력 → LLM → 응답               │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3: Post-processing                                    │
│                                                              │
│  1. Emotion.update() — 감정 트리거 포함 (별도 LLM 호출)        │
│  2. Memory.update()  — 감정 태그 포함 (별도 LLM 호출)          │
│  3. History.update() — 이번 턴 저장                           │
│  병렬 실행, 실패 시 롤백                                       │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
  응답 출력
```

## 5. 데이터 흐름 예시

```
사용자: "오늘 시험 봤는데 완전 망했어"

Stage 1 (Context Gathering):
├─ persona: 전체 페르소나
│   → 성격, 말투, 행동 지침, 내면 상태
├─ emotion: { "장난기": 0.7 }
├─ emotion_triggers: "시험" → 관련 트리거 없음
├─ knowledge.search_relevant("시험") → 관련 지식 없음
├─ fewshot.search("시험 망했어", emotions) → "위로" 태그 예시 2개
├─ memory.search("시험") → ["사용자는 대학생"]
├─ history.get_recent(5) → 최근 대화 5개

Stage 2 (PromptEngine 조립):
├─ [Identity] "당신은 유나입니다."
├─ [Persona] 성격, 말투, 가치관, 배경
├─ [Behavior] 상황별/주제별 행동 지침
│   "사용자가 고민을 털어놓을 때 → 경청하고 공감 먼저"
├─ [InnerWorld] 현재 생각, 숨기는 감정
├─ [Emotion] "장난기: 0.7"
├─ [Few-shot] 관련 예시 2개 (위로 상황)
├─ [Memory] "사용자는 대학생"
├─ [History] 최근 대화 5개
├─ [Response Guide] 응답 규칙
└─ → LLM 호출 → "아 정말? 시험 망했다니 속상하겠다 ㅠㅠ 어디가 어려웠어?"

Stage 3 (Post-processing):
├─ Emotion.update() → 장난기 감소, 공감 증가
│   (persona의 emotion_triggers와 연동)
├─ Memory.update() → "사용자가 시험을 망쳤다"
└─ History.update() → 턴 저장
```

## 6. 파일 구조

```
charactor_os/
├── spec/                          # 스펙 문서
│   ├── 00_architecture.md
│   ├── 01_persona.md             #
│   ├── 02_emotion.md
│   ├── 03_memory.md
│   ├── 04_knowledge.md           #
│   ├── 05_history.md
│   ├── 06_module_architecture.md
│   ├── 07_fewshot.md             # 신규
│   └── 08_prompt_engine.md       # 신규
├── src/
│   ├── character_os.py            # 통합 오케스트레이터
│   ├── api/
│   │   └── server.py              # FastAPI 서버
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── persona.py             # 확장
│   │   ├── emotion.py
│   │   ├── memory.py
│   │   ├── knowledge.py           # 확장
│   │   ├── history.py
│   │   └── fewshot.py             # 신규
│   └── prompts/
│       ├── templates.py           # 레거시 (점진적 제거)
│       └── engine.py              # 신규 (PromptEngine)
├── personas/                      # 페르소나 YAML
│   └── sample.yaml
├── knowledge/                     # 지식
│   ├── world.yaml
│   ├── relationships.yaml
│   ├── timeline.yaml
│   ├── characters/
│   └── freeform/
├── examples/                      # Few-shot 예시
│   ├── greeting.yaml
│   ├── comfort.yaml
│   └── ...
├── memory/                        # 영속화 (런타임)
├── frontend/                      # React 프론트엔드
└── main.py                        # 진입점
```

## 7. 기술 스택

| 컴포넌트 | 기술 |
|---|---|
| LLM | OpenAI API 또는 로컬 (mlx-lm) |
| 벡터 유사도 | numpy dot product |
| 임베딩 | sentence-transformers (all-MiniLM-L6-v2) |
| 영속화 | SQLite (기억), JSON (감정, 대화) |
| 프롬프트 | PromptEngine (동적 조립) |
| API | FastAPI (REST) |
| 프론트엔드 | React + Tailwind + shadcn/ui |

## 8. 검증 기준

| 검증 항목 | 기준 |
|---|---|
| 페르소나 로드 | 확장 YAML 파싱 → 모든 필드 접근 가능 |
| 행동 지침 | 상황별/주제별/절대 규칙이 프롬프트에 포함 |
| Few-shot 검색 | 관련 예시가 동적으로 선택됨 |
| Knowledge 구조화 | world/character/relationship/timeline/location 로드 |
| 프롬프트 토큰 예산 | 총 토큰이 상한을 초과하지 않음 |
| 프롬프트 동적 조립 | 관련성 기반으로 섹션이 선택적으로 포함 |
| 통합 대화 | 3-stage 파이프라인 전체 동작 → 일관된 캐릭터 응답 |
| 호환 | 기존 persona YAML도 정상 동작 |
