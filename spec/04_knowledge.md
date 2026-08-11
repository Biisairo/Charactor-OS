# Knowledge 모듈 스펙

## 1. 목적

캐릭터가 속한 **세계관, 관계, 타임라인, 장소** 등 구조화된 지식을 관리한다.
Persona의 배경을 넘어서는 세계 전체의 맥락을 제공한다.

## 2. Knowledge vs Persona 구분

| 구분 | Persona | Knowledge |
|---|---|---|
| 범위 | 캐릭터 자체의 정체성 | 캐릭터가 속한 세계 전체 |
| 예시 | "홍길동은 서자다" | "조선시대는 서자가 차별받는다" |
| 관리 | 캐릭터 1개당 1 파일 | 세계 1개당 N 파일 |
| 참조 | 시스템 프롬프트에 항상 포함 | 관련성에 따라 선택적 포함 |

## 3. Knowledge YAML 스키마

기존 단순 파일 로드도 계속 지원하되, `.yaml`/`.yml` 파일은 아래 스키마로 파싱하여 구조화한다.

### 3.1 World (세계관)

```yaml
# knowledge/world.yaml
type: world
name: string                    # 세계 이름 (예: "조선시대 세계관")
era: string                     # 시대/에포크 (예: "조선 중기")
description: string             # 세계관 요약

rules:                          # 세계의 규칙/법칙
  - string                      # 예: "서자는 과거를 볼 수 없다"

technology_level: string        # 기술 수준 (예: "전근대", "근대")
social_structure: string        # 사회 구조 (예: "양반-상민-천민 신분제")
```

### 3.2 Character (타 캐릭터)

```yaml
# knowledge/characters/아버지.yaml
type: character
name: string                    # 캐릭터 이름
identity: string                # 한 줄 소개
personality: string             # 성격 요약
relationship_to_player: string  # 주인공과의 관계
status: string                  # 현재 상태 (예: "생존", "행방불명")
first_appearance: string        # 첫 등장 시점
description: string             # 상세 설명
```

### 3.3 Relationship (관계 그래프)

```yaml
# knowledge/relationships.yaml
type: relationships
relationships:
  - from: string                # 출발 캐릭터 (예: "홍길동")
    to: string                  # 대상 캐릭터 (예: "아버지")
    type: string                # 관계 유형 (예: "부자", "연인", "적")
    sentiment: string           # 감정 (예: "분노", "그리움", "무관심")
    description: string         # 관계 설명
    strength: float             # 관계 강도 (0.0~1.0)
```

### 3.4 Timeline (타임라인)

```yaml
# knowledge/timeline.yaml
type: timeline
events:
  - time: string                # 시점 (예: "1592년", "어린 시절", "3일 전")
    event: string               # 사건 설명
    characters_involved: string[]  # 관련 캐릭터
    impact: string              # 영향/결과
```

### 3.5 Location (장소)

```yaml
# knowledge/locations.yaml
type: locations
locations:
  - name: string                # 장소 이름
    description: string         # 장소 묘사
    significance: string        # 스토리적 의미
    characters_present: string[] # 이 장소에 있는 캐릭터
```

### 3.6 Freeform (자유 형식 — 기존 호환)

```yaml
# knowledge/freeform/notes.yaml 또는 .md/.json/.txt
type: freeform                  # 또는 type 필드 생략
content: |
  임의의 텍스트...
```

## 4. 디렉토리 구조

```
knowledge/
├── world.yaml                  # 세계관 (type: world)
├── relationships.yaml          # 관계 그래프 (type: relationships)
├── timeline.yaml               # 타임라인 (type: timeline)
├── locations.yaml              # 장소 (type: locations)
├── characters/                 # 타 캐릭터 (type: character)
│   ├── father.yaml
│   └── friend.yaml
└── freeform/                   # 자유 형식 지식
    ├── notes.md
    └── facts.json
```

## 5. KnowledgeModule 클래스

### 인터페이스

```python
class KnowledgeModule:
    def __init__(self, knowledge_dir: str):
        """지식 디렉토리를 설정한다."""

    def load_all(self) -> None:
        """지식 디렉토리의 모든 파일을 로드하여 구조화/비구조화 데이터를 저장한다."""

    def to_prompt(self) -> str:
        """전체 지식을 프롬프트 문자열로 변환한다. (기존 호환)"""

    def get_world(self) -> dict | None:
        """세계관 정보를 반환한다."""

    def get_characters(self) -> list[dict]:
        """타 캐릭터 목록을 반환한다."""

    def get_character(self, name: str) -> dict | None:
        """특정 캐릭터 정보를 반환한다."""

    def get_relationships(self) -> list[dict]:
        """관계 그래프를 반환한다."""

    def get_relationships_for(self, character: str) -> list[dict]:
        """특정 캐릭터 관련 관계만 반환한다."""

    def get_timeline(self) -> list[dict]:
        """타임라인 이벤트를 반환한다."""

    def get_locations(self) -> list[dict]:
        """장소 목록을 반환한다."""

    def search_relevant(self, query: str, token_budget: int = 500) -> str:
        """쿼리와 관련된 지식만 선택하여 프롬프트 문자열로 반환한다."""
```

### search_relevant() 로직

1. 쿼리에서 키워드 추출
2. 각 knowledge 항목과 키워드 매칭 (이름, 설명, 사건 등)
3. 관련성 높은 순으로 정렬
4. token_budget 내에서 선택
5. 프롬프트 문자열로 조립

### to_prompt() 출력 형식 (기존 호환)

```
[캐릭터 지식]
--- world.yaml ---
{조선시대 세계관 요약}

--- characters/father.yaml ---
{아버지 정보}

--- relationships.yaml ---
{관계 그래프}

--- timeline.yaml ---
{타임라인}
```

### get_world() 프롬프트 변환

```
[세계관: 조선시대 세계관]
시대: 조선 중기
{description}
규칙:
- 서자는 과거를 볼 수 없다
- 양반-상민-천민 신분제
```

### get_characters() 프롬프트 변환

```
[등장인물]
- 아버지: {identity}. {personality}. 홍길동과의 관계: {relationship}
- 친구: {identity}. {personality}.
```

### get_relationships_for("홍길동") 프롬프트 변환

```
[관계]
- 홍길동 → 아버지: 부자 관계, 분노. {description}
- 홍길동 → 친구: 친구, 신뢰. {description}
```

## 6. 검증 기준

| 검증 | 기준 |
|---|---|
| world 로드 | type:world YAML → get_world() 반환 |
| character 로드 | type:character YAML → get_characters() 포함 |
| relationship 로드 | type:relationships YAML → get_relationships() 반환 |
| timeline 로드 | type:timeline YAML → get_timeline() 반환 |
| location 로드 | type:locations YAML → get_locations() 반환 |
| freeform 호환 | 기존 .md/.json/.txt 파일 정상 로드 |
| search_relevant | 키워드 매칭 → 관련 지식만 반환 |
| 빈 디렉토리 | to_prompt() → "지식 없음" |
| 하위 디렉토리 | characters/, freeform/ 하위 파일 스캔 |
