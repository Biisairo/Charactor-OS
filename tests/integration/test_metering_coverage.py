"""계측 누락 검출 (TASK-10).

"모든 LLM 호출이 로그에 남는가"는 코드를 눈으로 훑어서 답할 수 없다.
호출 지점이 추가되었는데 감싸지 않으면 집계가 조용히 틀리고, 그 사실을
아무도 모른다.

**원본 클라이언트가 스스로 호출 수를 센다.** 그 값과 계측된 값이 다르면
어딘가 프록시를 우회한 호출이 있다는 뜻이다. 새 호출 지점이 추가되어도
이 테스트가 자동으로 잡는다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.llm.client import TokenUsage, TrimmedMessage
from tests.conftest import PipelineMockClient, make_character_os


class CountingClient(PipelineMockClient):
    """자기가 실제로 몇 번 불렸는지 세는 클라이언트.

    프록시를 거치든 안 거치든 최종적으로 여기까지 온다. 따라서 이 카운트가
    호출의 '진실'이고, 계측값과 비교하면 누락을 찾을 수 있다.
    """

    def __init__(self, response: str = "흠, 반갑소."):
        super().__init__(response=response)
        self.raw_calls = 0

    def call_llm(self, messages, tools=None, use_stream=False, mute=True, **kwargs):
        self.raw_calls += 1
        result = super().call_llm(messages, tools=tools, use_stream=use_stream, mute=mute, **kwargs)
        return TrimmedMessage(
            content=result.content,
            role=result.role,
            reasoning_content=result.reasoning_content,
            tool_calls=result.tool_calls,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        )


class TestEveryCallIsMetered:
    @pytest.mark.parametrize("no_review", [True, False], ids=["reflection-off", "reflection-on"])
    def test_chat_covers_all_calls(self, character_dir: Path, tmp_path: Path, no_review: bool):
        client = CountingClient()
        cos = make_character_os(character_dir, tmp_path, client, no_review=no_review)

        cos.chat("안녕하시오")

        assert cos._meter.summary()["calls"] == client.raw_calls, (
            "계측되지 않은 LLM 호출이 있다 — 새 호출 지점을 _meter.wrap()으로 감쌌는지 확인하라"
        )

    def test_chat_stream_covers_all_calls(self, character_dir: Path, tmp_path: Path):
        client = CountingClient()
        cos = make_character_os(character_dir, tmp_path, client)

        list(cos.chat_stream("안녕하시오"))

        assert cos._meter.summary()["calls"] == client.raw_calls

    def test_memory_conflict_check_is_metered(self, character_dir: Path, tmp_path: Path):
        """기억 충돌 판정은 조건부 호출이라 누락되기 쉽다."""
        client = CountingClient()
        cos = make_character_os(character_dir, tmp_path, client)

        # 두 턴을 돌려 기존 기억이 생긴 뒤 충돌 판정 경로가 열리게 한다
        cos.chat("나는 한양에서 주막을 하오")
        client.raw_calls = 0
        cos.chat("장사가 통 되질 않소")

        assert cos._meter.summary()["calls"] == client.raw_calls

    def test_failed_call_is_counted(self, character_dir: Path, tmp_path: Path):
        """실패한 호출도 원본에는 도달했으므로 계측에 잡혀야 한다."""
        client = CountingClient()
        cos = make_character_os(character_dir, tmp_path, client)

        original = client.call_llm
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                client.raw_calls += 1  # 원본에 도달한 것으로 센다
                raise RuntimeError("프로바이더 거부")
            return original(*args, **kwargs)

        client.call_llm = flaky
        cos.chat("안녕하시오")

        summary = cos._meter.summary()
        assert summary["calls"] == client.raw_calls
        assert summary["failed_calls"] == 1


class TestLabelsCoverKnownSites:
    def test_all_known_labels_appear(self, character_dir: Path, tmp_path: Path):
        """알려진 호출 지점이 모두 라벨로 나타나야 한다."""
        client = CountingClient()
        cos = make_character_os(character_dir, tmp_path, client, no_review=False)

        cos.chat("안녕하시오")

        assert set(cos._meter.summary()["by_label"]) == {
            "response",
            "reflection",
            "emotion",
            "memory",
        }

    def test_no_unlabeled_calls(self, character_dir: Path, tmp_path: Path):
        client = CountingClient()
        cos = make_character_os(character_dir, tmp_path, client, no_review=False)

        cos.chat("안녕하시오")

        assert all(r.label for r in cos._meter.records)
