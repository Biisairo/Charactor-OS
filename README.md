# Character OS

[![CI](https://github.com/dongyo12/charactor_os/actions/workflows/ci.yml/badge.svg)](https://github.com/dongyo12/charactor_os/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-175%20passing-brightgreen)

> 감정·기억·지식을 가진 캐릭터가 일관된 인격으로 대화하는 **LLM 에이전트 런타임**.
>
> "프롬프트에 페르소나를 넣는다"가 아니라, **상태를 가진 6개 모듈을 3단계 파이프라인으로 오케스트레이션**하고,
> 제한된 토큰 예산 안에서 무엇을 프롬프트에 넣을지 매 턴 결정한다.

---

## 동작 예시

> 아래는 `--trace` 출력의 **형식 예시**입니다. 실제 실행 결과로 교체하세요.

```
사용자: 오늘 시험 완전히 망했어

[Stage 1] context gathering        12ms
  ├ memory.search("시험")          → 1건 (score 0.61, 임계값 0.3 통과)
  ├ fewshot.search(...)            → "위로" 태그 2건
  └ knowledge.search_relevant(...) → 0건

[Stage 2] response generation     1,840ms
  ├ prompt assembled               2,847 / 3,000 tokens
  ├ draft                          → LLM
  └ reflection: 말투 불일치 감지   → 1회 재생성

캐릭터: 아이고, 그리 상심 말게. 나도 과거엔 숱하게 무너졌네...

[Stage 3] post-processing           420ms
  ├ emotion  { 연민 0.4 → 0.7 }
  ├ memory   + "사용자가 시험을 망침"
  └ history  + 1턴
```

> **스크린샷 자리** — `docs/images/` 에 웹 UI 캡처를 추가하세요.

---

## 왜 이 프로젝트인가

LLM 캐릭터 챗봇의 어려움은 모델 호출이 아니라 **상태 관리와 컨텍스트 선택**에 있습니다.
이 프로젝트는 그 문제들을 하나씩 설계 결정으로 풀어낸 결과물입니다.

| 문제 | 이 프로젝트의 해법 |
|---|---|
| 대화가 길어지면 프롬프트가 무한히 커진다 | **PromptEngine 토큰 예산제** — 고정 상한 3,000 토큰, Always 섹션 우선 확보 후 잔여분을 비율 배분 |
| 관련 없는 기억이 프롬프트를 오염시킨다 | **관련성 임계값** — 코사인 유사도 0.3 미만은 top-k 안이어도 버림 |
| LLM이 캐릭터 말투를 이탈한다 | **Reflection 패턴** — 초안을 스스로 검토하고 재생성. 상한 2회로 비용·지연 제한 |
| 후처리 중 실패하면 상태가 깨진다 | **스냅샷 기반 롤백** — 감정·기억·히스토리를 원자적으로 되돌린 뒤 예외 전파 |
| 에이전트가 왜 그렇게 답했는지 알 수 없다 | **PipelineTrace** — Stage별 소요 시간·토큰·모듈 기여도 기록, `GET /api/trace/last` |
| LLM 호출 없이는 테스트가 불가능하다 | **클라이언트 의존성 주입** — API 키 없이 175개 테스트가 2초 내 완주 |

---

## 아키텍처

```
                    사용자 입력
                        │
    ┌───────────────────▼───────────────────────────────────┐
    │ Stage 1 — Context Gathering                           │
    │   정적:  Persona · Knowledge · FewShot                │
    │   동적:  Emotion · Memory · History                   │
    │   → 각 모듈이 입력 관련성에 따라 컨텍스트를 반환        │
    └───────────────────┬───────────────────────────────────┘
                        │
    ┌───────────────────▼───────────────────────────────────┐
    │ Stage 2 — Response Generation                         │
    │   PromptEngine: 토큰 예산 내에서 시스템 프롬프트 조립  │
    │   LLM 호출 → 초안                                      │
    │   ReflectionReviewer: 검토 → (필요 시) 재생성 ≤2회     │
    └───────────────────┬───────────────────────────────────┘
                        │
    ┌───────────────────▼───────────────────────────────────┐
    │ Stage 3 — Post-processing (스냅샷 → 실패 시 롤백)      │
    │   emotion.update() ∥ history.add()   ← 병렬            │
    │   memory.update()                    ← 감정 결과 의존  │
    │   영속화 (SQLite / JSON)                               │
    └───────────────────┬───────────────────────────────────┘
                        ▼
                      응답
```

세부 설계와 트레이드오프는 **[ARCHITECTURE.md](ARCHITECTURE.md)**, 모듈별 명세는 **[spec/](spec/)** 에 있습니다.

### 모듈

| 모듈 | 종류 | 책임 |
|---|---|---|
| `PersonaModule` | 정적 | 성격·말투·행동 지침·감정 트리거·내면 상태 |
| `KnowledgeModule` | 정적 | 세계관·관계 그래프·타임라인·장소 |
| `FewShotModule` | 정적 | 예시 대화 검색 (태그 + 임베딩 + 감정) |
| `EmotionModule` | 동적 | 0~1 스케일 감정 상태, 트리거 기반 갱신, 감쇠 |
| `MemoryModule` | 동적 | 벡터 검색 기반 장기 기억, 충돌 감지, 가중치 관리 |
| `HistoryModule` | 동적 | 최근 N턴 대화 기록 |
| `PromptEngine` | — | 토큰 예산 기반 동적 프롬프트 조립 |
| `ReflectionReviewer` | — | 응답 자기 검토 및 개선 루프 |

**정적 모듈은 에이전트가 절대 수정하지 않는다**는 것이 핵심 불변식입니다.
캐릭터의 정체성은 사람이 정의하고, 에이전트는 그 위에서 감정과 기억만 축적합니다.

---

## 빠른 시작

### 요구사항
- Python 3.10+, [uv](https://docs.astral.sh/uv/)
- Node.js 22+ (웹 UI를 쓸 경우)
- OpenAI API 키 (테스트 실행에는 **불필요**)

### 설치

```bash
git clone https://github.com/dongyo12/charactor_os.git
cd charactor_os
uv sync
cp .env.example .env      # OPENAI_API_KEY 입력
```

### 테스트 (API 키 없이 동작)

```bash
uv run pytest -q
# 175 passed in 2.27s
```

### CLI 대화

```bash
uv run python main.py
uv run python main.py --trace       # 파이프라인 트레이싱 출력
uv run python main.py --no-review   # Reflection 비활성화 (비용 절감)
uv run python main.py --debug       # 모듈별 상세 로그
```

### 웹 UI

```bash
cd frontend && npm ci && npm run build && cd ..
uv run uvicorn src.api.server:app --reload
# http://localhost:8000        웹 UI
# http://localhost:8000/docs   Swagger
```

### 캐릭터 추가

`characters/<이름>/` 아래에 YAML만 넣으면 됩니다. 코드 수정은 필요 없습니다.

```
characters/hong-gil-dong/
├── persona.yaml            # 성격, 말투, 감정 트리거, 내면 상태
├── knowledge/
│   ├── world.yaml          # 세계관
│   ├── relationships.yaml  # 관계 그래프
│   ├── timeline.yaml       # 연표
│   └── locations.yaml      # 장소
└── examples/               # few-shot 예시 (태그별)
    ├── greeting.yaml
    ├── comfort.yaml
    └── ...
```

```bash
uv run python main.py --character characters/my-character
```

---

## 기술 스택

| 영역 | 선택 | 이유 |
|---|---|---|
| LLM | OpenAI API / 로컬 (mlx-lm) | 동일 인터페이스로 교체 가능 |
| 임베딩 | sentence-transformers (all-MiniLM-L6-v2) | 로컬 실행, 외부 호출 0 |
| 벡터 검색 | numpy dot product | 기억 규모(수백 건)에 벡터 DB는 과설계 |
| 영속화 | SQLite (기억) / JSON (감정·대화) | 의존성 없이 단일 파일로 재현 가능 |
| API | FastAPI + WebSocket | 토큰 단위 스트리밍 |
| 프론트엔드 | React 19 + Tailwind 4 + shadcn/ui | — |
| 품질 | pytest · ruff · GitHub Actions | — |

---

## 프로젝트 구조

```
src/
├── character_os.py       # 3-stage 오케스트레이터
├── trace.py              # 파이프라인 트레이싱
├── prompts/engine.py     # 토큰 예산 기반 프롬프트 조립
├── modules/              # 6개 캐릭터 모듈 + Reflection
├── llm/                  # LLM 클라이언트 (API / 로컬)
└── api/server.py         # FastAPI REST + WebSocket

tests/
├── unit/                 # 모듈 단위 (8개 파일)
└── integration/          # 파이프라인 · API

spec/                     # 모듈별 설계 명세 9종
frontend/                 # React 웹 UI
characters/               # 캐릭터 정의 (YAML)
```

---

## 라이선스

MIT
