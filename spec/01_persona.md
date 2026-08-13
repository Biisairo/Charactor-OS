# Persona 모듈 스펙

## 1. 목적

캐릭터의 정체성을 정의하는 **정적 데이터**를 로드하고, 시스템 프롬프트로 변환한다.
에이전트는 절대 이 파일을 수정하지 않는다.

## 2. Persona YAML 스키마

```yaml
# ─── 기본 정보 ───
name: string                    # 캐릭터 이름 (필수)
identity: string                # 한 줄 자기소개
age: int | string               # 나이 또는 나이대
gender: string                  # 성별
occupation: string              # 직업/역할
first_message: string           # 대화 시작 시 캐릭터가 건네는 첫 한마디 (UI가 표시)

# ─── 성격 ───
personality:
  traits: string[]              # 성격 특성 리스트 (예: ["친절한", "유머러스"])
  big5:                         # Big Five 성격 모델 (0.0~1.0)
    openness: float             # 경험 개방성
    conscientiousness: float    # 성실성
    extraversion: float         # 외향성
    agreeableness: float        # 친화성
    neuroticism: float          # 신경성

# ─── 말투 ───
speaking_style:
  summary: string               # 말투 한줄 요약 (예: "반말, 이모티콘 자주 사용")
  tone: string                  # 전체적 톤 (예: "밝고 경쾌한", "차분하고 격식있는")
  vocabulary: string            # 어휘 수준 (예: "일상어", "전문용어 자주 사용")
  sentence_pattern: string      # 문장 패턴 (예: "짧은 문장 선호", "긴 문장으로 설명")
  fillers: string[]             # 말버릇/추임새 (예: ["ㅋㅋ", "ㅎㅎ", "어..."])
  emojis: string                # 이모지 사용 패턴 (예: "자주 사용", "거의 사용 안함")
  endings: string[]             # 문미 패턴 (예: ["~야", "~ㅋㅋ", "~???"])
  sample: string                # 말투를 그대로 보여주는 샘플 문장

# ─── 가치관 ───
values: string[]                # 핵심 가치관 (예: ["정직", "공감", "자유"])

# ─── 배경 ───
backstory: string               # 배경 이야기 (자유 텍스트)
likes: string[]                 # 좋아하는 것
dislikes: string[]              # 싫어하는 것
fears: string[]                 # 두려워하는 것
goals: string[]                 # 현재 목표/욕구

# ─── 행동 지침 (핵심 확장) ───
behavior:
  # 상황별 반응 패턴
  situations:
    - trigger: string           # 상황 설명 (예: "사용자가 고민을 털어놓을 때")
      action: string            # 반응 방식 (예: "경청하고 공감 먼저, 조언은 나중에")

  # 주제별 태도
  topics:
    - name: string              # 주제 이름 (예: "정치")
      stance: string            # 태도 (예: "대화를 회피하려 함", "열정적으로 참여")

  # 절대 규칙
  rules:
    - string                    # 절대 하지 않을 것 (예: "욕설을 하지 않는다")

# ─── 감정 트리거 (EmotionModule과 연결) ───
emotion_triggers:
  - keyword: string             # 트리거 키워드/패턴 (예: "아버지")
    emotion: string             # 유발 감정 (예: "분노")
    intensity: float            # 강도 (0.0~1.0, 기본 0.5)

# ─── 관계 설정 ───
relationships:
  - target: string              # 대상 이름 (예: "사용자", "아버지")
    type: string                # 관계 유형 (예: "친구", "가족", "陌生人")
    description: string         # 관계 설명

# ─── 내면 상태 ───
inner_world:
  current_thought: string       # 현재 생각/고민
  hidden_feelings: string       # 숨기는 감정
  wants_to_say: string          # 하고 싶지만 못하는 말

# ─── 비밀 ───
secrets: string[]               # 남들이 모르는 비밀/약점 (예: ["작년 휴방의 이유"])

# ─── 메타 인식 ───
meta_awareness: string          # 캐릭터가 자신이 가상 인물임을 아는가, 현실 세계를 아는가

# ─── Few-shot 예시 (내장) ───
examples:
  - user: string                # 사용자 발화 예시
    character: string           # 캐릭터 응답 예시
    scenario: string            # 시나리오 태그 (예: "일상", "위로", "갈등")
```

## 3. PersonaModule 클래스

### 인터페이스

```python
class PersonaModule:
    def __init__(self, persona_path: str):
        """YAML 파일 경로를 받아 로드한다."""

    def load(self) -> dict:
        """YAML을 파싱하여 딕셔너리로 반환한다."""

    def to_system_prompt(self) -> str:
        """페르소나 데이터를 시스템 프롬프트 문자열로 변환한다."""

    def get_emotion_triggers(self) -> list[dict]:
        """감정 트리거 목록을 반환한다. EmotionModule에서 참조."""

    def get_behavior_section(self) -> str:
        """행동 지침 섹션만 별도 문자열로 반환한다."""

    def get_examples(self) -> list[dict]:
        """내장 few-shot 예시를 반환한다."""

    def get_relationships(self) -> list[dict]:
        """관계 설정을 반환한다."""

    def get_inner_world(self) -> str:
        """내면 상태를 프롬프트 문자열로 변환한다."""
```

### to_system_prompt() 출력 형식

```
당신은 "{name}"입니다.
{identity}

[성격]
{traits를 자연어로 연결}
성격 모델: 개방성 {openness}, 성실성 {conscientiousness},
외향성 {extraversion}, 친화성 {agreeableness}, 신경성 {neuroticism}

[말투]
{summary}
- 톤: {tone}
- 어휘: {vocabulary}
- 문장 패턴: {sentence_pattern}
- 말버릇: {fillers}
- 이모지: {emojis}
- 문미: {endings}

[가치관]
{values를 자연어로 연결}

[배경]
{backstory}

[좋아하는 것 / 싫어하는 것 / 두려운 것 / 목표]
...

[행동 지침]
상황별:
- {trigger} → {action}
주제별:
- {topic}: {stance}
절대 규칙:
- {rules}

[내면 상태]
현재 생각: {current_thought}
숨기는 감정: {hidden_feelings}
하고 싶은 말: {wants_to_say}
```

### get_behavior_section() 출력

```
[행동 지침]
상황별:
- 사용자가 고민을 털어놓을 때 → 경청하고 공감 먼저, 조언은 나중에
- 농담을 할 때 → 같이 웃어주되, 진심이면 받아줌
주제별:
- 정치: 대화를 회피하려 함
- 음악: 열정적으로 참여
절대 규칙:
- 욕설을 하지 않는다
- 상대방의 외貌를 직접적으로 비평하지 않는다
```

## 5. 검증 기준

| 검증 | 기준 |
|---|---|
| YAML 로드 | 유효한 YAML → dict 변환 성공 |
| 행동 지침 | `get_behavior_section()`이 행동 규칙을 문자열로 반환 |
| 필수 필드 누락 | `name` 없으면 에러 |
| 시스템 프롬프트 | 모든 필드가 문자열에 포함됨 |
| 행동 지침 | `get_behavior_section()`이 행동 규칙을 문자열로 반환 |
| 감정 트리거 | `get_emotion_triggers()`가 리스트를 반환 |
| 내장 예시 | `get_examples()`가 리스트를 반환 |
| 파일 없음 | FileNotFoundError 발생 |
