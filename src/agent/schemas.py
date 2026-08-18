"""뇌가 주고받는 값들 (SPEC-09 §5.2).

`ThoughtBundle`이 Stage 1과 Stage 2 사이의 유일한 계약이다. 이전에는
`ContextBundle`이 그 자리에 있었지만 아무도 읽지 않았고, 실제 컨텍스트는
`PromptEngine`이 모듈을 다시 뒤져서 만들었다. 그래서 "뇌가 무엇을 근거로
그렇게 판단했는가"와 "프롬프트에 무엇이 들어갔는가"가 서로 달랐다.
이제 번들에 담긴 것만 프롬프트가 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

THOUGHT_KINDS = ("question", "hypothesis")


@dataclass(frozen=True)
class ResponseStrategy:
    """뇌가 세운 응답 전략. 발화가 아니라 발화의 방침이다."""

    situation: str = ""
    intent: str = ""
    avoid: str = ""
    tone: str = ""

    def is_empty(self) -> bool:
        """상한에 걸려 종료하면 전략이 없다. 없는 것을 지어내지 않는다."""
        return not any((self.situation, self.intent, self.avoid, self.tone))


@dataclass(frozen=True)
class NewThought:
    """작업기억에 새로 올릴 미해결 사고. id와 턴 번호는 모듈이 부여한다."""

    kind: str
    content: str
    confidence: float = 0.0


@dataclass(frozen=True)
class ToolCallRecord:
    """도구 호출 1건의 기록. 트레이스가 "왜 그렇게 판단했는가"를 재구성하는 재료다."""

    iteration: int
    name: str
    arguments: dict
    result_len: int = 0
    duration_ms: float = 0.0
    error: str = ""


@dataclass
class ThoughtBundle:
    """뇌의 산출물 — 수집한 근거와 그 위에서 세운 전략."""

    strategy: ResponseStrategy
    # 도구 없이 뇌가 이미 알던 것. Stage 2도 같은 것을 봐야 한다 (REQ-RA-73).
    baseline: dict[str, str] = field(default_factory=dict)
    collected: dict[str, str] = field(default_factory=dict)
    iterations: int = 0
    hit_cap: bool = False
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    resolved_ids: list[str] = field(default_factory=list)
    new_thoughts: list[NewThought] = field(default_factory=list)

    def tools_used(self) -> list[str]:
        return sorted(self.collected)
