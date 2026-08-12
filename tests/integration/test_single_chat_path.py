"""대화 경로 단일화 (TASK-11, REQ-11-1 · REQ-11-2 · REQ-11-6).

이전에는 대화 경로가 둘이었다. 웹 UI가 쓰는 `chat_stream()`에는 Reflection
검토가 없어, 검토를 거치지 않은 응답이 사용자에게 도달했다.

검토를 우회하는 경로가 다시 생기면 이 테스트가 실패한다. 그것이 이 파일의 목적이다.

이 테스트는 LLM API 키 없이 결정론적으로 통과한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.character_os import CharacterOS
from src.llm.client import TrimmedMessage
from tests.conftest import MockClient, make_character_os

SRC_DIR = Path(__file__).resolve().parents[2] / "src"


class RoutingClient(MockClient):
    """프롬프트 내용에 따라 다른 응답을 돌려주는 더블.

    검토기는 "PASS"로 시작하는 응답을 통과로 해석한다.
    """

    def __init__(self, response: str = "흠, 반갑구나."):
        super().__init__(response=response)
        self.labels: list[str] = []

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
        self.last_kwargs = {"use_stream": use_stream, "mute": mute}
        joined = " ".join(str(m.get("content", "")) for m in messages)

        if "검토" in joined or "PASS" in joined:
            self.labels.append("reflection")
            content = "PASS"
        else:
            self.labels.append("other")
            content = self._response.content

        return TrimmedMessage(
            content=content,
            role="assistant",
            reasoning_content="",
            tool_calls=[],
            usage=None,
        )


# ---------------------------------------------------------------------------
# REQ-11-1 — 검토를 우회하는 경로가 없다
# ---------------------------------------------------------------------------


class TestSinglePath:
    def test_chat_stream_is_removed(self):
        assert not hasattr(CharacterOS, "chat_stream"), (
            "검토를 우회하는 스트리밍 경로가 다시 생겼다 (TASK-11 참조)"
        )

    def test_streaming_response_generator_is_removed(self):
        assert not hasattr(CharacterOS, "_generate_response_streaming")

    def test_websocket_chat_endpoint_is_removed(self):
        from src.api.server import app

        paths = {getattr(route, "path", None) for route in app.routes}
        assert "/api/ws/chat" not in paths


# ---------------------------------------------------------------------------
# REQ-11-6 — 응답 생성은 비스트리밍 호출을 쓴다
# ---------------------------------------------------------------------------


class TestNonStreaming:
    def test_src_has_no_streaming_call(self):
        offenders = [
            f"{path.relative_to(SRC_DIR.parent)}:{lineno}"
            for path in SRC_DIR.rglob("*.py")
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if "use_stream=True" in line
        ]
        assert offenders == [], f"스트리밍 호출이 남아 있다: {offenders}"

    def test_response_generation_passes_use_stream_false(self, character_dir: Path, tmp_path: Path):
        client = RoutingClient()
        cos = make_character_os(character_dir, tmp_path / "state", client)
        cos.chat("안녕")

        assert client.last_kwargs is not None
        assert client.last_kwargs["use_stream"] is False


# ---------------------------------------------------------------------------
# REQ-11-2 — 대화가 검토를 거친다
# ---------------------------------------------------------------------------


class TestReflectionOnChatPath:
    def test_chat_invokes_reviewer(self, character_dir: Path, tmp_path: Path):
        client = RoutingClient()
        cos = make_character_os(character_dir, tmp_path / "state", client, no_review=False)
        cos.chat("안녕")

        assert "reflection" in client.labels, (
            "대화가 Reflection 검토를 거치지 않았다 (TASK-11 참조)"
        )

    @pytest.mark.anyio
    async def test_api_chat_invokes_reviewer(self, character_dir: Path, tmp_path: Path):
        """웹 UI가 쓰는 경로(`POST /api/chat`)도 검토를 거쳐야 한다."""
        from unittest.mock import AsyncMock, patch

        from httpx import ASGITransport, AsyncClient

        from src.api.server import app

        client = RoutingClient()
        cos = make_character_os(character_dir, tmp_path / "state", client, no_review=False)
        mock_run = AsyncMock(side_effect=lambda fn: fn())

        with (
            patch("src.api.deps.get_cos", return_value=cos),
            patch("src.api.deps.run_in_worker", mock_run),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as http:
                response = await http.post("/api/chat", json={"message": "안녕"})

        assert response.status_code == 200
        assert "reflection" in client.labels
