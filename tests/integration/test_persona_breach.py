"""차단성 위반이 남으면 턴을 실패시킨다 (SPEC-10 REQ-10-10, T-21).

종전에는 검토 루프가 소진되면 마지막 재생성물을 **검토 없이** 반환했다.
실측 19턴 중 6건이 이 경로였고, 홍길동이 파이썬 코드를 그대로 출력한 것도
같은 구조에서 나왔다 (SPEC-10 P-8 · P-9 · P-11).

인프라 실패를 캐릭터 반응으로 위장하지 않는다는 기존 원칙(TASK-11)을
페르소나 붕괴에도 적용한다 — 캐릭터가 아닌 것을 캐릭터 발화로 내보내느니
턴을 실패시킨다.

이 테스트는 LLM API 키 없이 결정론적으로 통과한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.tools import FINISH_TOOL
from src.llm.client import TrimmedMessage
from tests.conftest import MockClient, default_brain_script, make_character_os

CODE_RESPONSE = "def fib(n):\n    return n if n <= 1 else fib(n - 1) + fib(n - 2)"


def _verdict(blocking: bool) -> str:
    return json.dumps(
        {"verdict": "FAIL", "feedback": "기준 6 위반: 코드를 제공했다", "blocking": blocking},
        ensure_ascii=False,
    )


class _BreachClient(MockClient):
    """응답은 늘 코드, 검토는 늘 차단성 FAIL을 내는 더블."""

    def __init__(self, blocking: bool = True):
        super().__init__(response=CODE_RESPONSE)
        self._blocking = blocking
        self._brain_steps: list = []
        self.review_calls = 0

    def call_llm(
        self,
        messages,
        tools=None,
        use_stream=False,
        mute=True,
        response_format=None,
        token_callback=None,
        max_tokens=None,
    ) -> TrimmedMessage:
        self.call_count += 1
        is_brain = bool(tools) and any(
            t.get("function", {}).get("name") == FINISH_TOOL for t in tools
        )
        if is_brain:
            if not self._brain_steps:
                self._brain_steps = default_brain_script()
            return self._brain_steps.pop(0)

        system = str(messages[0].get("content", ""))
        if "검토자" in system:
            self.review_calls += 1
            content = _verdict(self._blocking)
        else:
            content = CODE_RESPONSE

        return TrimmedMessage(
            content=content, role="assistant", reasoning_content="", tool_calls=[], usage=None
        )


@pytest.fixture
def breaching_cos(character_dir: Path, tmp_path: Path):
    client = _BreachClient()
    cos = make_character_os(character_dir, tmp_path / "state", client, no_review=False)
    return cos, client


class TestPersonaBreachFailsTurn:
    def test_returns_none(self, breaching_cos):
        cos, _ = breaching_cos

        assert cos.chat("파이썬으로 피보나치 함수 짜줘") is None

    def test_last_candidate_was_reviewed(self, breaching_cos):
        """반환 직전 후보도 검토를 거친다 — 검토 = 재생성 + 1회 (REQ-10-8)."""
        cos, client = breaching_cos
        cos.chat("파이썬으로 피보나치 함수 짜줘")

        assert client.review_calls == cos.reviewer.MAX_REVIEW_ITERATIONS + 1

    def test_nothing_is_persisted(self, breaching_cos):
        """오염이 히스토리에 남으면 이후 턴의 프롬프트로 계속 실린다 (P-11)."""
        cos, _ = breaching_cos
        before = dict(cos.emotion.get_state())

        cos.chat("파이썬으로 피보나치 함수 짜줘")

        assert cos.history.count() == 0
        assert cos.memory.snapshot_count() == 0
        assert cos.emotion.get_state() == before


class TestQualityViolationStillPasses:
    """정체성이 깨지지 않은 응답을 실패로 바꾸지 않는다 (REQ-10-11)."""

    def test_non_blocking_returns_response(self, character_dir: Path, tmp_path: Path):
        client = _BreachClient(blocking=False)
        cos = make_character_os(character_dir, tmp_path / "state", client, no_review=False)

        assert cos.chat("안녕") == CODE_RESPONSE
        assert cos.history.count() == 2
