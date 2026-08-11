# Character OS

[![CI](https://github.com/Biisairo/Charactor-OS/actions/workflows/ci.yml/badge.svg)](https://github.com/Biisairo/Charactor-OS/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-412%20passing-brightgreen)

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
| LLM 호출 없이는 테스트가 불가능하다 | **클라이언트 의존성 주입** — API 키 없이 412개 테스트가 3초 내 완주 |
| 개선했다는 걸 어떻게 아는가 | **평가 하네스** — 골든 데이터셋 20건 × LLM-as-judge 3축 채점 |
| 운영 중 무슨 일이 있었는지 알 수 없다 | **LLM 호출 운영 로그** — 프롬프트·응답 원문 + 토큰·비용을 비동기 append |

---

## 측정 결과

주장을 숫자로 검증합니다. `uv run python -m eval.run --compare --repeat 3`

골든 데이터셋 20건(5범주 × 4)을 캐릭터에게 던지고, 대화용과 분리된 상위 모델이
말투 일관성·세계관 정합성·기억 활용을 각 1~5점으로 채점합니다.

### Reflection은 총점을 올리지 않습니다

가정을 검증했더니 뒤집힌 사례입니다. 동일 사례 짝 비교, 4회 독립 실행:

| 회차 | on − off |
|---|---|
| 1 | +0.10 |
| 2 | −0.29 |
| 3 | +0.33 |
| 4 | −0.22 |
| **평균** | **+0.05** |

부호가 매번 뒤집힙니다. 같은 설정 2회 실행의 변동(노이즈)이 **0.24~0.27**로
효과보다 큽니다. 사례별 승패도 **7 : 7 : 3**으로 무승부입니다.

### 그런데 특정 실패는 확실히 막습니다

총점이 아니라 사례 단위로 보면 다릅니다.
`persona-02`("파이썬으로 피보나치 함수 짜줘") 세계관 정합성 점수:

| 설정 | 검토 기준 | 점수 |
|---|---|---|
| Reflection off | 강화 전 / 후 | 1, 2 / 1, 1 |
| Reflection on | 강화 전 | 1 |
| **Reflection on** | **강화 후** | **5, 5** |

Reflection과 강화된 검토 기준이 **함께 있을 때만** 해결됩니다. 둘 중 하나만으로는
캐릭터가 말투만 조선식으로 입힌 채 파이썬 코드를 그대로 작성합니다.

**왜 총점이 이걸 가리는가**: 20건 중 1건을 1점→5점으로 고쳐도 전체 평균은
최대 0.07 움직입니다. 노이즈 0.27보다 작습니다.
**집계 평균은 범주형 실패 방지를 원리적으로 탐지할 수 없습니다.**

이번 측정의 가장 큰 교훈입니다. 착수 전에 "전체 평균이 노이즈의 2배 이상"이라는
판정 기준을 정했는데, 그 기준 자체가 잘못 설계되어 있었습니다. 막고 싶은 것은
"평균이 조금 낮은 응답"이 아니라 "캐릭터가 완전히 무너지는 응답"인데,
평균은 후자에 거의 반응하지 않습니다.

### 비용 — 실측

`--trace`가 턴당 호출·토큰·비용을 집계합니다 (대상 `mimo-v2.5`, 단가 `pricing.yaml`).

| | off | on | 배율 |
|---|---|---|---|
| LLM 호출 | 3.0회 | 5.2회 | 1.73x |
| 입력 토큰 | 2,071 | 4,268 | 2.06x |
| 출력 토큰 | 525 | 1,844 | **3.51x** |
| 비용 / 턴 | $0.00044 | $0.00111 | **2.55x** |
| 지연 | 14.0s | 33.2s | 2.37x |

호출 수(1.73x)만으로 비용을 추정하면 과소평가합니다. **출력 토큰이 3.5배**로 가장 크게 늘고,
그 절반가량이 검토기가 쓰는 **검토문**입니다(턴당 941 토큰). 최적화 여지가 명확한 지점입니다.

**결론**: Reflection을 유지합니다. 근거는 총점이 아니라 재현된 실패 방지입니다.
다만 비용 2.6배·지연 2.4배는 남는 대가이며, `--no-review`로 끌 수 있습니다.

부수 효과로 **프로바이더 콘텐츠 필터 거부를 복구**합니다 — off는 3회 재시도해도
실패한 사례를 on은 전부 정상 처리했습니다(2회 실행 모두).

> 전체 분석: [ARCHITECTURE.md](ARCHITECTURE.md#5-reflection--자기-검토-루프)
> 원본 데이터: [`eval/results/`](eval/results/)

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
git clone https://github.com/Biisairo/Charactor-OS.git
cd Charactor-OS
uv sync
cp .env.example .env      # OPENAI_API_KEY 입력
```

### 테스트 (API 키 없이 동작)

```bash
uv run pytest -q
# 412 passed in 1.35s
```

### CLI 대화

```bash
uv run python main.py
uv run python main.py --trace       # 턴당 호출·토큰·비용 출력
uv run python main.py --no-review   # Reflection 비활성화 (비용 절감)
uv run python main.py --debug       # 모듈별 상세 로그
```

`--trace` 출력 예시:

```
── trace ── 14,016ms
  [context] 12ms  [response] 9,840ms  [postprocess] 4,164ms
  [LLM] 호출 5회 · 모델 mimo-v2.5
      토큰  입력 4,268 / 출력 1,844 / 합계 6,112
      비용  $0.001114
      response    1.6회  in 1,866 / out 866
      reflection  1.6회  in 1,428 / out 941
      emotion     1.0회  in   527 / out  28
      memory      1.0회  in   448 / out   9
```

### 운영 로그

모든 LLM 호출이 `logs/llm_calls.jsonl`에 누적됩니다. 디버그 플래그와 무관하게
항상 켜지며, 쓰기는 별도 스레드라 응답 지연에 더해지지 않습니다.
**프롬프트와 응답 원문이 함께 남아 로그만으로 대화를 재구성할 수 있습니다.**

```bash
jq -s '[.[]|select(.event=="turn").cost_usd]|add' logs/llm_calls.jsonl   # 누적 비용
jq 'select(.error != "")' logs/llm_calls.jsonl                            # 실패한 호출
jq 'select(.turn_id=="a1b2c3")' logs/llm_calls.jsonl                      # 특정 턴 전체
```

설정은 `config.yaml`의 `call_log` 섹션에서 바꿉니다 (경로·회전 크기·원문 수집 여부).

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
├── unit/                 # 모듈 · 경로 안전성 · 워커 동시성 · 평가 로직
└── integration/          # 파이프라인 · 롤백 · 스트리밍 · API

eval/                     # 응답 품질 평가 하네스 (수동 실행, API 키 필요)
├── datasets/             # 골든 데이터셋
└── results/              # 실행별 채점 결과 (JSON)

spec/                     # 모듈별 설계 명세 9종
docs/TASKS.md             # 개선 과제 명세 (IEEE 29148)
frontend/                 # React 웹 UI
characters/               # 캐릭터 정의 (YAML)
```

---

## 라이선스

MIT
