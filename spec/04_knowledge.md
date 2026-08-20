# Knowledge 모듈 스펙

문서 버전: 3.1
최종 갱신: 2026-08-19 (TASK-21 — 검색을 순위+게이트로. 상세는 `spec/12_embedding.md`)

## 1. 목적

캐릭터가 속한 세계의 맥락을 관리한다. Persona가 "이 사람이 누구인가"라면
Knowledge는 "이 사람이 사는 세상이 어떻게 돌아가는가"다.

## 2. Knowledge vs Persona

| | Persona | Knowledge |
|---|---|---|
| 범위 | 캐릭터 자체의 정체성 | 캐릭터가 속한 세계 |
| 예 | "홍길동은 서자다" | "조선에서 서자는 과거를 볼 수 없다" |
| 파일 | 캐릭터당 1개 (`persona.yaml`) | 캐릭터당 N개 (마크다운) |

---

## 3. 두 종류의 지식

```
characters/<id>/static/knowledge/
    base/       배경지식 — 원문 그대로 프롬프트에 주입된다
    general/    일반지식 — RAG로 색인되어 검색으로 꺼내 쓴다
    *.md        루트 직속 파일은 general 취급 (하위 호환)
```

| | base/ | general/ |
|---|---|---|
| 프롬프트 | 원문 그대로 항상 | 검색된 조각만 |
| 크기 | 짧게 (`BASE_WARN_TOKENS` 초과 시 경고) | 제한 없음 |
| 역할 | 캐릭터의 상식 + 일반지식 인덱스 | 참고 자료 |
| 예 | 세계의 규칙, 늘 지키는 직업적 습관 | 연표, 장소, 관계, 업무 상세 |

### 배치는 사람이 정한다

자동 분류하지 않는다. 어느 지식이 "항상 알아야 하는 것"인지는 캐릭터를 만든
사람만 안다. 파일을 옮기는 것으로 성격이 바뀐다.

### base/ 는 파일명 순으로 실린다

주입 순서가 곧 읽는 순서다. 순서를 통제하려면 파일명에 번호를 붙인다.

```
base/01-world.md              세계가 어떻게 돌아가는가
base/02-broadcast-rules.md    직업적으로 늘 지키는 것
```

### 배경지식이 인덱스가 된다

`base/` 원문 뒤에 `general/`의 문서·소제목 목차가 **자동으로** 붙는다.
사람이 목차를 손으로 관리하면 자료를 추가할 때마다 어긋나고, 어긋난 목차는
캐릭터가 그 자료를 영영 찾지 못하게 만든다.

---

## 4. 파일 형식 — 마크다운 하나 (v3)

지식 파일은 **마크다운만** 쓴다. `.md`가 아닌 파일은 읽지 않는다.

### 왜 하나로 통일하는가

v2까지는 구조화 YAML(`type: world|relationships|timeline|locations`)과 자유
문서가 공존했다. 실제로 그 스키마가 지탱하던 기능을 전수 조사한 결과는 이렇다.

| 구조화 API | 실사용 |
|---|---|
| `get_world().era` | Reflection의 시대 기준, eval 판정자 프로필 — **실사용** |
| `get_relationships_for()` | 뇌의 `get_relationships` 도구 |
| `get_timeline` · `get_locations` · `get_relationships` | API에만 노출, 호출하는 곳 없음 |
| `get_characters` | 초기화 로그의 개수 세기뿐 |
| `to_prompt()` (전체 지식) | 호출하는 곳 없음 |

스키마 100여 줄이 지탱하던 실제 기능은 `era` 한 줄과 관계 조회 하나였다.
그 값에 비해 저작 비용이 크다 — 자료를 추가할 때마다 "이건 어느 스키마인가"를
판단해야 하고, 검색 단위가 청킹기 내부 규칙에 맡겨지며, YAML 문법이 그대로
프롬프트에 실려 노이즈가 된다.

마크다운 하나로 통일하면 **저작자가 `##` 로 검색 단위를 직접 통제**하고,
쓴 문장이 그대로 프롬프트에 실린다.

### front matter — 시스템이 읽는 최소 메타

`era`는 실제로 쓰이므로 남긴다. 문서 앞머리에 YAML front matter로 적는다.

```markdown
---
era: 2020년대 후반 대한민국 서울
---

# 이 바닥이 돌아가는 방식

개인방송은 실시간으로 나가지만…
```

- front matter는 **본문이 아니다**. 프롬프트에도 검색 색인에도 들어가지 않는다.
- 여러 파일에 있으면 파일명 순으로 먼저 발견된 값을 쓴다.
- 없으면 `era`는 빈 값이고, Reflection은 시대 정합성 기준을 검사하지 않는다
  (TASK-19 REQ-19-2).

### 관계는 어디에 적는가

- **사용자·주요 인물과의 관계** → `persona.yaml`의 `relationships`.
  시스템 프롬프트에 항상 실린다.
- **그 밖의 인물 관계** → `general/`의 마크다운. 검색으로 꺼내 쓴다.

`get_relationships` 도구는 폐지했다. `search_knowledge`가 같은 일을 하며,
도구가 하나 줄어 뇌의 루프도 줄어든다.

---

## 5. RAG 색인

### 청킹

1. 마크다운 제목(`#` ~ `######`)을 경계로 나눈다.
2. 조각이 `MAX_CHUNK_CHARS`(600)를 넘으면 **문단 단위로 한 번 더** 나눈다.
3. 각 조각은 상위 제목 경로를 머리에 달고 다닌다 — `방송 정보 > 시청자 호칭과 문화`.
   맥락 없이 잘린 문단은 검색되어도 무슨 이야기인지 알 수 없다.

저작자는 `##` 를 끊는 것으로 검색 단위를 직접 정한다. "이 내용은 따로 검색되면
좋겠다" 싶으면 제목을 하나 만들면 된다.

### 검색 — 키워드와 임베딩을 함께 쓴다

점수 = 키워드 매칭 + 임베딩 유사도 × `EMBEDDING_WEIGHT`(0.3).

임베딩만 쓰지 않는 이유가 있다. **모델은 순위를 잘 매기지만 "관련된 것이
아예 없다"는 판정을 못 한다.** 적중과 잡음의 유사도 분포가 겹치기 때문이며,
절대 임계값으로도 z-score 상대화로도 가를 수 없었다. 반면 키워드는 무관한
질의를 완벽히 막는다 — 대신 표현이 어긋난 질의를 전혀 잡지 못한다(top-1 6%).

그래서 임베딩을 두 가지로 제한해 섞는다.

| 장치 | 값 | 하는 일 |
|---|---|---|
| `EMBEDDING_TOP_K` | 2 | 유사도 상위 2개만 보너스를 받는다 |
| `NO_KEYWORD_GATE` | 0.85 | 키워드가 하나도 걸리지 않은 조각은 이 유사도를 넘어야 후보가 된다 |
| `EMBEDDING_WEIGHT` | 0.3 | 확실한 키워드 신호를 임베딩이 뒤집지 못하게 한다 |

게이트는 키워드 근거가 **없는** 조각에만 적용된다. 이미 근거가 있는 조각에서는
임베딩이 순위만 조정하면 된다.

이 조합에서 top-1 69%, 무관한 질의 무응답 100%다. 게이트를 빼면 top-1 이 75%로
오르지만 무관한 질의마다 조각 2개가 프롬프트에 실린다 — 예산이 빡빡한
상황에서 그 6%p(16건 중 1건)를 사려고 낼 값이 아니라고 판단했다.

**세 값 모두 현재 임베딩 모델의 유사도 분포에 매여 있다.** 모델을 바꾸면
`eval/embedding_probe.py --hybrid`를 다시 돌려 재조정해야 한다. 상세한 측정과
근거는 `spec/12_embedding.md` 4.2 에 있다.

`embedding_fn`이 없으면 키워드 점수만으로 동작한다 (FewShot과 같은 정책).

### 임베딩은 캐시된다

조각 임베딩은 `state/embeddings.db`에 콘텐츠 해시로 캐시된다. 파일을 옮기거나
이름을 바꿔도 살아남고, 문서의 한 조각만 고치면 그 조각만 다시 계산된다.
모델을 바꾸면 전량 미스가 되므로 별도의 무효화 절차가 없다 (SPEC-12).

---

## 6. 모듈 인터페이스

```python
class KnowledgeModule:
    def __init__(self, knowledge_dir: str, embedding_fn=None): ...

    def load_all(self) -> None: ...
    @property
    def load_issues(self) -> list[AssetLoadIssue]: ...

    def era(self) -> str:            # front matter에서. 없으면 ""
    def base_text(self) -> str:      # 배경지식 원문
    def to_index(self) -> str:       # 일반지식 목차
    def to_base_prompt(self) -> str: # 원문 + 목차 (뇌가 받는 형태)

    def chunks(self) -> list[KnowledgeChunk]: ...
    def search_relevant(self, query: str, token_budget: int = 500) -> str: ...
```

발화 단계(Stage 2)에는 `base_text()`만 간다. 목차는 "무엇을 더 찾아볼 수
있는가"이므로 검색을 결정하는 뇌에만 필요하다.

---

## 7. v2 → v3 마이그레이션

1. `world.yaml` → `base/01-world.md`. `era`는 front matter로, `description`과
   `rules`는 산문으로 옮긴다.
2. `relationships.yaml` → 사용자·주요 인물은 `persona.yaml`로, 나머지는
   `general/`의 마크다운으로.
3. `timeline.yaml` · `locations.yaml` → `general/`의 마크다운. 항목 하나가
   검색 단위가 되도록 `##` 로 끊는다.
4. 남은 `.yaml`은 읽히지 않는다. 지우거나 `.md`로 옮긴다.
