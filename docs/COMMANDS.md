# 명령 모음

이 저장소에서 실제로 쓰는 명령을 한곳에 모았다.
모든 명령은 **리포지토리 루트**에서 실행한다 (프론트엔드 절 제외).

**API 키가 필요한지**를 항목마다 표시했다. 키 없이 도는 것부터 익히면
`git clone` 직후에 바로 확인할 수 있다.

| 표시 | 뜻 |
|---|---|
| 🔑 | LLM API 키 필요 (`.env`) |
| 💸 | 실제 비용 발생 |
| — | 키 불필요 |

---

## 1. 설치

```bash
uv sync                      # Python 의존성
cp .env.example .env         # OPENAI_API_KEY 등 입력
cd frontend && npm install   # 웹 UI를 쓸 때만
```

`.env`는 두 벌로 나뉜다. **대화용과 평가 판정용은 서로 다른 모델을 쓴다.**

| 변수 | 용도 |
|---|---|
| `OPENAI_API_KEY` · `OPENAI_MODEL` · `OPENAI_BASE_URL` | 캐릭터 대화 |
| `EVAL_API_KEY` · `EVAL_MODEL` · `EVAL_BASE_URL` | 평가 판정자 (LLM-as-judge) |

---

## 2. 테스트·검사 — 키 불필요

```bash
env -u OPENAI_API_KEY uv run pytest -q      # 549개, 2초 내
```

`env -u OPENAI_API_KEY`를 붙이는 이유는 **키가 없어도 통과해야 한다는 것 자체가
불변식**이기 때문이다. 키가 있는 셸에서 그냥 돌리면 실수로 실제 호출이 섞여도
모른다. CI도 키 없이 돈다.

```bash
uv run pytest -q tests/unit                 # 단위만
uv run pytest -q tests/integration          # 통합만
uv run pytest -q -k "reflection"            # 이름으로 고르기
uv run pytest -q --lf                       # 직전에 실패한 것만
uv run pytest tests/unit/test_persona.py -v # 파일 하나, 상세
```

### 린트·포맷

```bash
uv run ruff check .                # 린트
uv run ruff check . --fix          # 자동 수정
uv run ruff format .               # 포맷 적용
uv run ruff format --check .       # 포맷 검사만 (CI가 쓰는 형태)
```

### 프론트엔드

```bash
cd frontend
npm run lint        # oxlint
npx tsc -b --force  # 타입체크만
npm run build       # tsc -b && vite build
npm run dev         # 개발 서버 (HMR)
```

> `npm run build`가 타입체크를 포함한다. `tsc -b`는 캐시를 쓰므로,
> 타입 오류를 확실히 다시 보려면 `--force`를 붙인다.

---

## 3. 대화 — 🔑

### CLI

```bash
uv run python main.py                                   # 기본 캐릭터 (홍길동)
uv run python main.py --character characters/han-so-min # 캐릭터 지정
uv run python main.py --trace                           # 턴마다 트레이스 출력
uv run python main.py --debug                           # 모듈별 상세 로그
uv run python main.py --no-review                       # Reflection 끄기 (비용·지연 절감)
uv run python main.py --config path/to/config.yaml      # 설정 파일 지정
```

종료는 `quit` 또는 `exit`.

**파이프로 자동 실행**할 수도 있다.

```bash
printf '안녕!\n뭐해?\nquit\n' | uv run python main.py --trace
```

> 파이프로 돌리면 프롬프트와 출력이 한 턴씩 어긋나 보인다. 응답만 뽑으려면
> `캐릭터:` 부터 `── trace ──` 직전까지를 잘라낼 것.

### 웹 UI

```bash
uv run uvicorn src.api.server:app --reload          # 개발 (자동 재시작)
uv run uvicorn src.api.server:app --port 8080       # 포트 지정
uv run python -m src.api.server --port 8080         # 동등, 자체 CLI
```

| 주소 | 내용 |
|---|---|
| http://localhost:8000 | 웹 UI |
| http://localhost:8000/docs | Swagger |

프론트엔드는 `frontend/dist`의 빌드 산출물을 서버가 서빙한다.
UI를 고쳤다면 `cd frontend && npm run build`를 먼저 돌릴 것.

---

## 4. 평가 — 🔑 💸

### 응답 품질 (LLM-as-judge)

**`--out`을 항상 줄 것.** 생략하면 `eval/results/`에 쓰면서
`*_latest.json`을 덮어쓴다. 실제로 `--dry-run`이 실측 결과를 더미 값으로
덮어쓴 적이 있다.

```bash
# 비용 0 — 실행 경로만 점검 (LLM 호출 없음)
uv run python -m eval.run --dry-run --out /tmp/evalout

# 한 설정만 (Reflection on) — 20건, 약 10분
uv run python -m eval.run --out eval/results/<주제>

# on/off 동시 비교 — 약 20~30분
uv run python -m eval.run --compare --out eval/results/<주제>

# 설정당 3회 반복 (노이즈 추정 포함) — 약 60~75분, ~$1.5
uv run python -m eval.run --compare --repeat 3 --out eval/results/<주제>

# 캐릭터 지정 / 앞 N건만
uv run python -m eval.run --character han-so-min --out /tmp/evalout
uv run python -m eval.run --limit 3 --out /tmp/evalout
```

`--compare`가 출력하는 것:

- 설정별 축 점수(말투·세계관·기억)와 범주별 평균
- **공통 사례 짝 비교** — 설정마다 채점 실패 수가 다르면 사례 집합이 어긋나므로,
  공통으로 채점된 사례만으로 차이를 낸다
- 사례별 승패 (on 우세 / off 우세 / 동률)
- 응답 지연 비교
- `--repeat 2` 이상이면 **실행 간 변동(노이즈 추정)**

> **`--repeat 1`(기본)로 결론을 내지 말 것.** 같은 설정 2회 실행의 노이즈가
> 0.24~0.27인데 on/off 차이는 그보다 작다(+0.05). 부호가 실행마다 뒤집힌다.
> 20건 표본으로는 0.5 미만의 총점 차이를 판별할 수 없다.

### few-shot 검색 정밀도 — 키 불필요

임베딩 모델이 로컬에서 돌기 때문에 API 키가 필요 없다.

```bash
PYTHONPATH=. uv run python -m eval.fewshot_probe                 # 정밀도 측정
PYTHONPATH=. uv run python -m eval.fewshot_probe --sweep         # 임계값 스윕
PYTHONPATH=. uv run python -m eval.fewshot_probe --json out.json # 결과 저장
```

`--sweep`은 `MIN_FEWSHOT_SCORE`를 바꿔가며 구간별 정확도를 비교한다.
임계값을 눈대중으로 정하지 않기 위한 도구다.

---

## 5. API

```bash
curl -s localhost:8000/api/health
curl -s localhost:8000/api/performance | jq       # 턴당 호출·토큰·비용
curl -s localhost:8000/api/trace/last | jq        # 직전 턴 트레이스

# 대화 (🔑)
curl -s -X POST localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"안녕하시오"}'

# 캐릭터 전환
curl -s -X POST localhost:8000/api/character/switch \
  -H 'Content-Type: application/json' \
  -d '{"character_id":"han-so-min"}'

# 상태 초기화 (history는 기본 false)
curl -s -X POST localhost:8000/api/character/reset \
  -H 'Content-Type: application/json' \
  -d '{"memory":true,"emotion":true,"history":true}'
```

<details>
<summary>전체 엔드포인트 (28개)</summary>

```
GET     /api/health
POST    /api/chat
GET     /api/trace/last
GET     /api/performance

GET     /api/emotion
GET     /api/memory
GET     /api/memory/stats
GET     /api/history
POST    /api/character/reset

GET     /api/characters
POST    /api/characters
DELETE  /api/characters/{character_id}
POST    /api/character/switch
GET     /api/persona
PUT     /api/persona

GET     /api/knowledge
GET     /api/knowledge/{name}
PUT     /api/knowledge/{name}
GET     /api/knowledge/locations
GET     /api/knowledge/relationships
GET     /api/knowledge/relationships/{character}
GET     /api/knowledge/timeline

GET     /api/fewshot
GET     /api/fewshot/search

GET     /api/logs
GET     /api/debug
POST    /api/debug/toggle
POST    /api/debug/clear
```

`/api/knowledge/{name}`은 리터럴 경로(`locations`·`relationships`·`timeline`)
**뒤에** 등록되어야 한다. 앞에 오면 리터럴 경로를 잠식한다 —
`tests/integration/test_api_surface.py`가 이 제약을 고정한다.

</details>

---

## 6. 운영 로그

모든 LLM 호출이 `logs/llm_calls.jsonl`에 append된다 (`config.yaml`의 `call_log`).

```bash
# 턴당 비용 합계
jq -s '[.[]|select(.event=="turn").cost_usd]|add' logs/llm_calls.jsonl

# 라벨별 토큰
jq -r 'select(.event=="call") | "\(.label) \(.prompt_tokens)/\(.completion_tokens)"' logs/llm_calls.jsonl

# 실패한 호출만
jq 'select(.error != "")' logs/llm_calls.jsonl

# 특정 턴 재구성
jq 'select(.turn_id=="<id>")' logs/llm_calls.jsonl
```

---

## 7. 상태 초기화

캐릭터의 동적 상태는 `characters/<id>/state/`에 있고 git이 추적하지 않는다.

```bash
rm -rf characters/hong-gil-dong/state    # 한 캐릭터
rm -rf characters/*/state                # 전부
```

서버가 떠 있다면 `POST /api/character/reset`을 쓰는 편이 낫다 —
인메모리 상태까지 함께 비운다.

---

## 8. 자주 겪는 함정

| 증상 | 원인·해결 |
|---|---|
| `ModuleNotFoundError: src` | 스크립트를 직접 실행할 때는 `PYTHONPATH=.`가 필요하다. `pytest`는 `pyproject.toml`에 설정되어 있어 불필요 |
| 코드를 되돌렸는데 테스트가 계속 실패 | `find . -name __pycache__ -prune -exec rm -rf {} +` |
| 평가 결과 파일이 더미 값으로 덮여 있음 | `--out` 없이 `--dry-run`을 돌린 것. `git checkout -- eval/results/`로 복원 |
| 캐릭터를 바꿨는데 이전 기억이 보임 | `config.yaml`에 `memory_db_path` 등을 명시하면 모든 캐릭터가 공유한다. 주석 처리하면 캐릭터별로 분리된다 |
| 프론트 수정이 웹 UI에 반영 안 됨 | 서버는 `frontend/dist`를 서빙한다. `npm run build` 필요 |
| `sed`의 `\b`가 안 먹음 | BSD(macOS) `sed`는 단어 경계를 지원하지 않는다. Python `re`를 쓸 것 |

---

## 9. CI가 돌리는 것

`.github/workflows/ci.yml` — push·PR 시 `main`에서 실행된다.

```bash
# backend (Python 3.13)
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -q          # API 키를 주입하지 않는다

# frontend (Node 22)
npm ci
npm run lint
npm run build
```

푸시 전에 로컬에서 같은 것을 돌려두면 배지가 빨개지는 일이 줄어든다.
