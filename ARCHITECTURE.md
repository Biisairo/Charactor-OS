# Architecture

이 문서는 Character OS가 **왜 지금의 구조인지**를 설명합니다.
무엇을 만들었는지는 [README](README.md)와 [spec/](spec/)에 있고, 여기서는 선택과 트레이드오프를 다룹니다.

---

## 1. 문제 정의

캐릭터 챗봇을 "긴 시스템 프롬프트 + LLM 호출"로 만들면 세 가지가 곧바로 무너집니다.

1. **컨텍스트 폭발** — 세계관·기억·대화 기록을 전부 넣으면 프롬프트가 턴마다 커지고, 비용과 지연이 선형으로 증가합니다.
2. **일관성 붕괴** — 프롬프트가 길어질수록 모델은 앞부분(말투 규칙)을 무시합니다.
3. **상태 부재** — 감정과 기억이 없으면 캐릭터는 매 턴 초기화됩니다.

Character OS는 이 셋을 각각 **토큰 예산제**, **Reflection**, **모듈화된 상태 관리**로 다룹니다.

---

## 2. 3-Stage 파이프라인

단일 LLM 호출 대신 세 단계로 나눈 이유는 **각 단계의 실패 특성이 다르기** 때문입니다.

| Stage | 실패 시 영향 | 대응 |
|---|---|---|
| 1. Context Gathering | LLM 호출 없음. 컨텍스트가 비어도 대화는 가능 | 관련 없으면 그냥 생략 (graceful degradation) |
| 2. Response Generation | 응답 자체가 없음 → 치명적 | 예외를 잡아 `None` 반환, 상태는 건드리지 않음 |
| 3. Post-processing | 응답은 이미 나감. 상태만 오염 | **스냅샷 롤백** 후 예외 전파 |

Stage 2가 실패하면 Stage 3은 아예 실행되지 않습니다. 사용자가 응답을 받지 못한 턴이
기억과 감정에 남는 일을 구조적으로 막습니다.

### Stage 3의 순서 제약

```
emotion.update()  ─┐
                   ├─ 병렬 (서로 독립)
history.add()     ─┘
        │
        ▼
memory.update(emotion_state)   ← 감정 결과에 의존, 순차
        │
        ▼
persist (SQLite / JSON)        ← 실패해도 롤백 불필요 (아직 미저장)
```

감정과 히스토리는 서로를 참조하지 않으므로 `ThreadPoolExecutor(max_workers=2)`로 묶었습니다.
기억은 "이 대화가 어떤 감정이었는가"를 태그로 저장하므로 감정 결과를 기다립니다.

영속화를 마지막에 둔 것이 롤백을 단순하게 만듭니다. 인메모리 상태만 되돌리면 되고,
디스크는 애초에 건드리지 않았으므로 부분 저장 상태가 존재하지 않습니다.

```python
emotion_snap  = self.emotion.snapshot()
memory_count  = self.memory.snapshot_count()
history_count = self.history.count()
try:
    ...
except Exception:
    self.emotion.restore(emotion_snap)
    self.memory.pop_last_n(self.memory.snapshot_count() - memory_count)
    self.history.pop_last_n(self.history.count() - history_count)
    raise
```

> **트레이드오프**: 카운트 기반 롤백은 "추가만 발생한다"는 가정에 의존합니다.
> Stage 3이 기존 기억을 수정하도록 확장하면 이 방식은 깨지고, 기억 단위 저널이 필요해집니다.
> 현재 범위에서는 단순함을 택했습니다.

---

## 3. PromptEngine — 토큰 예산제

가장 중요한 설계 결정입니다. **프롬프트 크기는 대화 길이와 무관하게 상한이 있어야 합니다.**

### 배분 알고리즘

```
1. Always 섹션을 먼저 확보한다 (양보 불가)
     Persona + Emotion + ResponseGuide + History(최근 10턴)

2. 잔여 = MAX_PROMPT_TOKENS(3000) − Always

3. 잔여를 관련성 기반 섹션에 비율 배분
     knowledge      35%
     fewshot        25%
     memory         20%
     relationships  10%
     history        10%

4. 각 모듈이 자기 예산 안에서 스스로 잘라 반환한다
```

**핵심은 3단계가 아니라 4단계입니다.** 엔진이 텍스트를 받아서 자르는 게 아니라,
`token_budget`을 인자로 넘기면 각 모듈이 자기 도메인 지식으로 무엇을 버릴지 결정합니다.
`KnowledgeModule`은 관련성 낮은 항목을 통째로 빼고, `FewShotModule`은 예시 개수를 줄입니다.
엔진이 일괄로 문자열을 자르면 YAML 구조가 중간에서 깨집니다.

### 왜 이 비율인가

Knowledge에 가장 큰 몫을 준 것은, 캐릭터가 무너지는 가장 흔한 원인이
"말투 이탈"이 아니라 **세계관 모순**이기 때문입니다. 말투는 Always 섹션의 Persona가
이미 보장하지만, 세계관은 매 턴 검색해서 넣어야 합니다.

Persona와 Emotion을 Always에 둔 것도 같은 이유입니다. 이 둘이 빠지면
캐릭터가 아니라 그냥 어시스턴트가 됩니다. 반면 few-shot이 빠져도 캐릭터는 유지됩니다.

> **알려진 한계**: 토큰 추정은 문자 수 기반 근사입니다. 정확한 tokenizer(tiktoken)를 쓰지 않은 것은
> 의존성을 늘리지 않기 위해서지만, 한국어에서는 오차가 큽니다. 실제 상한을 엄격히 지켜야 한다면
> 교체해야 할 지점입니다.

---

## 4. Memory — 관련성 임계값

초기 구현은 top-k 검색이었고, 이게 실제 대화에서 문제를 일으켰습니다.
기억이 3개뿐일 때 "안녕"이라고만 해도 **관련 없는 기억 3개가 전부 프롬프트에 들어갔습니다.**
top-k는 "k개를 채운다"는 뜻이지 "관련 있는 것만 준다"는 뜻이 아닙니다.

```python
MIN_RELEVANCE_SCORE = 0.3   # 코사인 유사도

# 점수 필터링 → 정렬 → top-k  (순서가 중요)
```

필터를 top-k **앞에** 둡니다. 뒤에 두면 k개를 뽑은 다음 걸러내므로
관련 있는 기억이 k 바깥에 있을 때 놓칩니다.

결과가 0건이면 빈 문자열이 아니라 "관련 기억 없음"을 명시합니다.
섹션 자체를 없애면 모델이 이전 턴의 기억을 환각으로 채우는 경향이 있었습니다.

**벡터 DB를 쓰지 않은 이유**: 기억 규모가 캐릭터당 수백 건입니다.
384차원 numpy dot product는 이 규모에서 마이크로초 단위이고, SQLite 한 파일로
전체 상태가 재현됩니다. pgvector나 Chroma는 운영 복잡도만 늘립니다.
규모가 10⁵을 넘으면 그때 교체할 지점입니다.

---

## 5. Reflection — 자기 검토 루프

Stage 2는 단순 LLM 호출이 아니라 생성 → 검토 → 개선 루프입니다.

```
draft = LLM(system_prompt, user_input)
for i in range(MAX_REVIEW_ITERATIONS):      # = 2
    result = reviewer.review(draft)          # 말투 · 감정 톤 · 금지 표현
    if result.passed:
        break
    draft = regenerate(result.feedback)
return draft
```

### 상한이 2인 이유

Reflection의 위험은 무한 루프가 아니라 **비용과 지연**입니다.
검토가 매번 실패하면 한 턴에 LLM을 5회 호출하게 됩니다(초안 1 + 검토 2 + 재생성 2).
실측에서 개선의 대부분은 1회차에 발생했고 2회차 이후는 수렴하지 않았습니다.
그래서 상한을 낮게 고정하고, 실패해도 **마지막 초안을 그대로 반환**합니다.
완벽한 응답보다 응답이 있는 편이 낫습니다.

`--no-review` 플래그로 전체를 끌 수 있습니다. 개발·테스트 중에는 이쪽이 기본값입니다.

---

## 6. 옵저버빌리티 — PipelineTrace

에이전트 시스템의 디버깅 난이도는 "왜 이렇게 답했는가"에 답할 수 없다는 데서 옵니다.
`PipelineTrace`는 Stage별 `StageTrace`를 모으고, 각 Stage는 자기 `details`에
모듈별 기여도를 기록합니다.

```
PipelineTrace
├── user_input
└── stages: [StageTrace]
        ├── name, started_at, finished_at, duration_ms
        └── details: { 모듈별 소요 시간, 출력 크기, 토큰, reflection 횟수 }
```

- CLI: `--trace`
- API: `GET /api/trace/last`

트레이싱은 기본 비활성입니다. 켜고 끄는 비용이 없어야 프로덕션 경로에 남길 수 있습니다.

---

## 7. 테스트 전략 — 의존성 주입

`CharacterOS.__init__`은 `client` 인자를 받습니다. 주입하면 `model_type`을 무시합니다.

```python
cos = CharacterOS(character_dir=..., client=MockClient())
```

이것 하나로 얻는 것:

- **API 키 없이 전체 파이프라인 테스트** — `git clone` 직후 `uv run pytest`가 통과합니다.
  CI에도 시크릿을 넣지 않습니다.
- **결정론적 테스트** — 175개가 2초 내에 끝납니다. LLM 응답을 고정하므로 재현 가능합니다.
- **실패 경로 검증** — `mock_client.call_llm`이 예외를 던지게 해서 Stage 3 롤백을 실제로 검증합니다.
  실제 LLM으로는 재현하기 어려운 경로입니다.

임베딩도 같은 방식입니다. `embedding_fn`을 주입받으므로 테스트에서는
seed 기반 더미 벡터를 씁니다 — sentence-transformers 모델 로드(수백 MB)를 건너뜁니다.

```
tests/
├── unit/           8개 모듈, 격리 테스트
└── integration/    3-stage end-to-end · 롤백 · 스트리밍 · REST API
```

---

## 8. API 계층 — CharacterWorker

`CharacterOS`는 스레드 안전하지 않습니다. 감정·기억·히스토리가 공유 가변 상태이고,
Stage 3은 여러 모듈을 순서대로 갱신합니다. 동시 요청이 이 사이를 끼어들면 상태가 찢어집니다.

락을 거는 대신 **전용 스레드 + 큐**를 택했습니다.

```
HTTP 요청 (async) ──▶ Queue ──▶ CharacterWorker 스레드 (순차 실행) ──▶ Future
```

- 모든 대화가 하나의 스레드에서 **순차 실행**되므로 락이 필요 없습니다.
- FastAPI의 이벤트 루프는 블로킹되지 않습니다 (`loop.call_soon_threadsafe`로 결과 전달).
- 상태 **읽기**(`GET /api/emotion` 등)는 큐를 거치지 않습니다. GIL 하에서 dict 읽기는
  원자적이고, 약간 오래된 값을 읽는 것은 허용 가능합니다.

WebSocket 스트리밍은 워커가 토큰을 `Queue`에 넣고, 이벤트 루프가
`run_in_executor`로 꺼내 전송합니다.

> **확장 시 한계**: 캐릭터 하나당 스레드 하나이므로 동시 사용자가 늘면 대화가 직렬화됩니다.
> 멀티 유저로 가려면 세션별 `CharacterOS` 인스턴스 + 워커 풀이 필요하고,
> 그 시점에는 상태 영속화도 파일에서 DB로 옮겨야 합니다.

---

## 9. 정적/동적 분리라는 불변식

| | 정적 | 동적 |
|---|---|---|
| 모듈 | Persona, Knowledge, FewShot | Emotion, Memory, History |
| 소유자 | 사람 (YAML 편집) | 에이전트 (런타임 갱신) |
| 영속화 | 리포지토리의 YAML | SQLite / JSON (gitignore) |

**에이전트는 정적 파일을 절대 수정하지 않습니다.**

이 규칙이 없으면 캐릭터가 자기 페르소나를 재작성하면서 서서히 표류합니다.
정체성은 고정하고 경험만 쌓게 하는 것 — 이게 "캐릭터가 일관되게 유지된다"의 실제 구현입니다.

부수 효과로 캐릭터 추가가 코드 변경 없이 디렉토리 하나로 끝납니다.

---

## 10. 앞으로

현재 구조에서 자연스럽게 이어지는 다음 단계들입니다.

- **토큰 추정 정확화** — 문자 기반 근사 → tiktoken
- **멀티 유저** — 세션별 인스턴스 + 워커 풀, 상태 저장소 DB 이전
- **기억 요약(compaction)** — 오래된 기억을 병합해 검색 품질 유지
- **평가 하네스** — 페르소나 일관성을 정량 측정 (현재는 Reflection의 정성 판단에 의존)
- **`api/server.py` 라우터 분리** — 단일 파일 800줄, 도메인별 `APIRouter`로 분리
