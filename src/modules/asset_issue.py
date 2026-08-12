"""정적 자산 로드 중 발생한 문제의 기록 (TASK-06).

정적 자산(persona · knowledge · examples)의 파싱 실패를 `except Exception: pass`로
무시하면, 프롬프트 품질이 **원인 불명으로** 떨어진다. YAML 문법 오류 하나로
few-shot 예시가 0개가 되어도 사용자도 개발자도 이를 알 수 없다.

모듈은 문제를 이 타입으로 모으기만 하고, 출력은 오케스트레이터가 맡는다.
모듈에 출력 콜백을 심지 않는 이유는 두 가지다.

- 모듈이 순수하게 유지되어 로드 결과를 값으로 검사할 수 있다 (테스트 가능).
- 어디에 어떻게 보일지는 오케스트레이터의 관심사다.

`expected`가 이 타입의 핵심이다. **의도된 폴백과 예기치 않은 실패는 다르다** (REQ-06-3).
knowledge의 구조화 파싱 실패는 freeform 처리라는 설계된 동작이고,
examples의 파싱 실패는 고쳐야 할 결함이다. 둘을 같은 말로 기록하면
로그를 봐도 무엇이 문제인지 알 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetLoadIssue:
    """자산 파일 하나를 읽는 중 생긴 문제."""

    filename: str
    reason: str
    # True면 설계된 폴백(예: knowledge의 freeform 처리),
    # False면 고쳐야 할 실패(예: examples YAML 문법 오류)
    expected: bool = False

    def describe(self) -> str:
        kind = "폴백" if self.expected else "로드 실패"
        return f"[{kind}] {self.filename}: {self.reason}"
