"""Stage 3의 프로바이더 거부 처리 (TASK-12, REQ-12-6).

TASK-11은 `response` 라벨의 거부만 막았다. `emotion`·`memory`는 거부 문자열을
받으면 JSON 파싱에 실패하고 조용한 폴백으로 넘어간다 — 감정이 갱신되지 않고
기억이 저장되지 않았는데도 아무 흔적이 남지 않는다.

Stage 3의 거부는 응답 자체를 망치지는 않으므로 **재시도하지 않는다**.
다만 조용히 삼키지 않는다. 계측이 거부를 표시하고, 운영 로그에 남는다.

이 테스트는 LLM API 키 없이 결정론적으로 통과한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.llm.client import TrimmedMessage
from tests.conftest import MockClient, make_character_os

REFUSAL = "The request was rejected because it was considered high risk"
VALID_RESPONSE = "흠, 그리 말하니 반갑구나."


class Stage3RefusingClient(MockClient):
    """응답 생성은 정상, 후처리(감정·기억)만 거부하는 더블."""

    def call_llm(
        self,
        messages,
        tools=None,
        use_stream=False,
        mute=True,
        response_format=None,
        token_callback=None,
    ) -> TrimmedMessage:
        self.call_count += 1
        self.last_messages = messages
        # 응답 생성만 페르소나 시스템 프롬프트를 싣는다.
        # 후처리(감정·기억)는 분석기 프롬프트를 쓴다.
        is_response = any(
            m.get("role") == "system" and "홍길동" in str(m.get("content", "")) for m in messages
        )

        return TrimmedMessage(
            content=VALID_RESPONSE if is_response else REFUSAL,
            role="assistant",
            reasoning_content="",
            tool_calls=[],
            usage=None,
        )


@pytest.fixture
def cos_with_refusing_stage3(character_dir: Path, tmp_path: Path):
    client = Stage3RefusingClient()
    cos = make_character_os(character_dir, tmp_path / "state", client, trace=True)
    return cos, client


class TestStage3RefusalIsVisible:
    def test_response_still_succeeds(self, cos_with_refusing_stage3):
        """후처리 거부가 응답을 망쳐서는 안 된다."""
        cos, _ = cos_with_refusing_stage3
        assert cos.chat("안녕") == VALID_RESPONSE

    def test_refusal_is_counted_in_metrics(self, cos_with_refusing_stage3):
        cos, _ = cos_with_refusing_stage3
        cos.chat("안녕")

        summary = cos._meter.summary()
        assert summary["refused_calls"] >= 1, (
            "후처리의 프로바이더 거부가 집계에 드러나지 않는다 (REQ-12-6)"
        )

    def test_refusal_appears_in_trace(self, cos_with_refusing_stage3):
        from src.trace import format_trace

        cos, _ = cos_with_refusing_stage3
        cos.chat("안녕")

        assert "거부" in format_trace(cos._last_trace)

    def test_clean_turn_reports_no_refusal(self, character_dir: Path, tmp_path: Path):
        from tests.conftest import PipelineMockClient

        client = PipelineMockClient(response=VALID_RESPONSE)
        cos = make_character_os(character_dir, tmp_path / "state", client, trace=True)
        cos.chat("안녕")

        assert cos._meter.summary()["refused_calls"] == 0
