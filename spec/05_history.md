# History 모듈 스펙

## 1. 목적

캐릭터와 사용자 간의 대화 기록을 관리한다.
ReAct loop가 필요할 때 최근 N개 대화를 조회할 수 있도록 한다.

**핵심 요구사항**: 대화 기록은 영속화되어야 한다. 프로그램을 재시작해도 이전 대화가 유지되어야 한다.

## 2. 대화 구조

```python
@dataclass
class ConversationTurn:
    role: str  # "user" | "character"
    content: str  # 대화 내용
    timestamp: float  # 시각 (time.time())
```

## 3. HistoryModule 클래스

### 인터페이스

```python
class HistoryModule:
    def __init__(self, save_path: str | None = None, max_turns: int = 100):
        """저장 경로와 최대 보관 턴 수를 설정한다."""

    def add_turn(self, role: str, content: str) -> None:
        """대화 한 턴을 추가한다."""

    def get_recent(self, n: int = 10) -> list[ConversationTurn]:
        """최근 n개 대화를 반환한다."""

    def to_prompt(self, n: int = 10) -> str:
        """최근 n개 대화를 프롬프트 문자열로 변환한다."""

    def save(self) -> None:
        """대화 기록을 JSON 파일로 저장한다."""

    def load(self) -> None:
        """JSON 파일에서 대화 기록을 로드한다."""
```

### to_prompt() 출력 형식

```
[최근 대화]
사용자: 안녕
캐릭터: 안녕! 반가워 😊
사용자: 오늘 뭐 했어?
캐릭터: 카페에서 코딩했어! 커피 맛있었음 ㅎㅎ
```

대화가 없으면:
```
[최근 대화]
대화 없음
```

## 4. 영속화

```
memory/history.json
{
    "turns": [
        {
            "role": "user",
            "content": "안녕",
            "timestamp": 1691234567.0
        },
        {
            "role": "character",
            "content": "안녕! 반가워 😊",
            "timestamp": 1691234568.0
        }
    ]
}
```

## 5. 검증 기준

| 검증 | 기준 |
|---|---|
| 턴 추가 | add_turn → get_recent에 포함 |
| 최근 N개 | max_turns 초과 시 오래된 것 제거 |
| 프롬프트 변환 | 대화가 문자열로 정확히 변환됨 |
| 영속화 | save → load 후 대화 복원 |
| **재시작 유지** | **프로그램 재시작 후 이전 대화가 유지됨** |
