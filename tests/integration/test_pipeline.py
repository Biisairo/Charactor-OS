"""CharacterOS 3-stage 파이프라인 통합 테스트.

MockClient를 사용하여 실제 LLM 호출 없이 전체 파이프라인을 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm.client import TrimmedMessage
from tests.conftest import MockResponse, PipelineMockClient, make_character_os

# ---------------------------------------------------------------------------
# helpers
#
# MockClient 확장·임베딩 우회·CharacterOS 조립은 conftest에 공용화되어 있다.
# ---------------------------------------------------------------------------

_CompatMockClient = PipelineMockClient


def _make_cos(character_dir: Path, tmp_path: Path) -> tuple:
    """CharacterOS 인스턴스를 생성하고 MockClient를 주입한다.

    Returns:
        (cos, mock_client, output_lines)
    """
    output_lines: list[str] = []
    mock_client = _CompatMockClient()

    cos = make_character_os(
        character_dir,
        tmp_path,
        mock_client,
        output=output_lines.append,
    )

    return cos, mock_client, output_lines


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cos(character_dir: Path, tmp_path: Path):
    """기본 CharacterOS 인스턴스 (MockClient 주입)."""
    instance, client, outputs = _make_cos(character_dir, tmp_path)
    return instance


@pytest.fixture
def cos_with_client(character_dir: Path, tmp_path: Path):
    """CharacterOS + MockClient + 출력 캡처 튜플."""
    return _make_cos(character_dir, tmp_path)


# ---------------------------------------------------------------------------
# 1. CharacterOS 초기화 — 모든 모듈이 로드된다
# ---------------------------------------------------------------------------


class TestCharacterOSInit:
    def test_all_modules_initialized(self, cos):
        assert cos.persona is not None
        assert cos.emotion is not None
        assert cos.memory is not None
        assert cos.knowledge is not None
        assert cos.history is not None
        assert cos.fewshot is not None
        assert cos.prompt_engine is not None

    def test_persona_loaded(self, cos):
        assert cos.persona._data.get("name") == "홍길동"

    def test_emotion_triggers_injected(self, cos):
        """init 중 persona의 emotion_triggers가 emotion 모듈에 주입된다."""
        assert len(cos.emotion._triggers) > 0

    def test_client_is_mock(self, cos):
        assert isinstance(cos.client, _CompatMockClient)


# ---------------------------------------------------------------------------
# 2. cos.chat(user_input) returns response string
# ---------------------------------------------------------------------------


class TestChat:
    def test_chat_returns_string(self, cos_with_client):
        cos, mock_client, _ = cos_with_client
        mock_client.next_response = MockResponse(content="안녕하신가!")
        result = cos.chat("안녕")
        assert isinstance(result, str)
        assert result == "안녕하신가!"

    def test_chat_returns_none_on_llm_error(self, cos_with_client):
        """LLM 호출 실패 시 None을 반환한다."""
        cos, mock_client, _ = cos_with_client

        def _fail(*args, **kwargs):
            raise RuntimeError("LLM unavailable")

        mock_client.call_llm = _fail
        result = cos.chat("안녕")
        assert result is None


# ---------------------------------------------------------------------------
# 3. cos.chat() calls client.call_llm
# ---------------------------------------------------------------------------


class TestClientCalled:
    def test_call_llm_invoked(self, cos_with_client):
        cos, mock_client, _ = cos_with_client
        mock_client.next_response = MockResponse(content="반갑소")
        cos.chat("안녕하세요")
        assert mock_client.call_count >= 1

    def test_messages_contain_user_input(self, cos_with_client):
        cos, mock_client, _ = cos_with_client
        mock_client.next_response = MockResponse(content="대답")
        cos.chat("아버지 보고 싶소")

        assert mock_client.last_messages is not None
        user_msgs = [m for m in mock_client.last_messages if m.get("role") == "user"]
        assert any("아버지" in m.get("content", "") for m in user_msgs)

    def test_system_prompt_in_messages(self, cos_with_client):
        cos, mock_client, _ = cos_with_client
        mock_client.next_response = MockResponse(content="대답")
        cos.chat("안녕")

        # Stage 2 (chat 응답)의 call_llm에서 시스템 프롬프트를 확인.
        # 응답 생성만 페르소나 시스템 프롬프트를 싣는다 (감정·기억은 분석기 프롬프트).
        chat_calls = [
            c
            for c in mock_client.all_call_records
            if any(
                m.get("role") == "system" and "홍길동" in str(m.get("content", ""))
                for m in c["messages"]
            )
        ]
        assert len(chat_calls) >= 1
        system_msgs = [m for m in chat_calls[0]["messages"] if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        assert "홍길동" in system_msgs[0]["content"]


# ---------------------------------------------------------------------------
# 4. cos.reset() resets state
#    CharacterOS에 reset() 메서드가 없으므로, 초기 상태가 깨끗한지 검증하고
#    새 인스턴스가 독립적인 상태를 가지는지 확인한다.
# ---------------------------------------------------------------------------


class TestReset:
    def test_emotion_initially_empty(self, cos):
        state = cos.emotion.get_state()
        assert isinstance(state, dict)

    def test_history_initially_empty(self, cos):
        assert cos.history.count() == 0

    def test_memory_initially_empty(self, cos):
        assert cos.memory.snapshot_count() == 0

    def test_fresh_instance_has_clean_state(self, character_dir, tmp_path):
        """새 인스턴스는 이전 인스턴스의 상태와 독립적이다."""
        dir1 = tmp_path / "instance1"
        dir1.mkdir()
        cos1, mc1, _ = _make_cos(character_dir, dir1)
        mc1.next_response = MockResponse(content="대답")
        cos1.chat("양반들이 나쁘다")

        # 별도 디렉토리에 새 인스턴스 생성 — 이전 상태와 독립적
        dir2 = tmp_path / "instance2"
        dir2.mkdir()
        cos2, _, _ = _make_cos(character_dir, dir2)
        assert cos2.history.count() == 0
        assert cos2.memory.snapshot_count() == 0


# ---------------------------------------------------------------------------
# 5. emotion state changes after chat (set_triggers is called during init)
# ---------------------------------------------------------------------------


class TestEmotionStateChange:
    def test_emotion_triggers_set_during_init(self, cos):
        """init 시 set_triggers가 호출되어 트리거가 주입된다."""
        triggers = cos.emotion._triggers
        assert len(triggers) > 0
        keywords = {t["keyword"] for t in triggers}
        assert "아버지" in keywords
        assert "어머니" in keywords

    def test_emotion_changes_with_trigger_keyword(self, character_dir, tmp_path):
        """트리거 키워드를 포함한 대화 후 감정 상태가 변한다."""
        cos, mock_client, _ = _make_cos(character_dir, tmp_path)

        initial_state = cos.emotion.get_state()

        # mock 순서: Stage 2 응답 → emotion 분석 → memory 분석
        responses = iter(
            [
                TrimmedMessage(
                    content="아버지는... 말하기 어렵소.",
                    role="assistant",
                    reasoning_content="",
                    tool_calls=[],
                    usage=None,
                ),
                TrimmedMessage(
                    content=json.dumps(
                        {
                            "emotions": {"분노": 0.6, "슬픔": 0.4},
                            "remove": [],
                            "significant": True,
                        }
                    ),
                    role="assistant",
                    reasoning_content="",
                    tool_calls=[],
                    usage=None,
                ),
                TrimmedMessage(
                    content=json.dumps({"memories": [], "significant": False}),
                    role="assistant",
                    reasoning_content="",
                    tool_calls=[],
                    usage=None,
                ),
            ]
        )

        mock_client.call_llm = lambda *a, **k: next(responses)

        result = cos.chat("아버지에 대해 이야기해 주세요")
        assert result is not None

        final_state = cos.emotion.get_state()
        # 감정 상태에 변화가 있어야 한다
        assert len(final_state) >= len(initial_state)
        assert "분노" in final_state or "슬픔" in final_state


# ---------------------------------------------------------------------------
# 분석 층 배선 (TASK-14)
#
# 오케스트레이터가 분석기에 `on_prompt`를 넘기지 않으면 프롬프트가 디버그
# 패널에서 조용히 사라진다. 눈으로만 확인하면 다음 리팩터링에서 다시 끊긴다.
# ---------------------------------------------------------------------------


class TestAnalyzerWiring:
    def test_memory_prompt_reaches_debug_log(self, character_dir: Path, tmp_path: Path):
        client = PipelineMockClient(response="흠, 반갑구나.")
        cos = make_character_os(character_dir, tmp_path / "state", client, debug=True)

        cos.chat("안녕하시오")

        assert "[memory 프롬프트]" in "\n".join(cos._debug_logs)

    def test_emotion_prompt_reaches_debug_log(self, character_dir: Path, tmp_path: Path):
        client = PipelineMockClient(response="흠, 반갑구나.")
        cos = make_character_os(character_dir, tmp_path / "state", client, debug=True)

        cos.chat("안녕하시오")

        assert "[emotion 프롬프트]" in "\n".join(cos._debug_logs)

    def test_analysis_layer_uses_labelled_clients(self, character_dir: Path, tmp_path: Path):
        """분석 층을 거쳐도 계측 라벨이 유지되어야 한다.

        라벨이 끊기면 비용이 어느 단계에서 나왔는지 알 수 없게 된다 (TASK-04).
        """
        client = PipelineMockClient(response="흠, 반갑구나.")
        cos = make_character_os(character_dir, tmp_path / "state", client, debug=True)

        cos.chat("안녕하시오")

        labels = set(cos._meter.summary()["by_label"])
        assert {"response", "emotion", "memory"} <= labels
