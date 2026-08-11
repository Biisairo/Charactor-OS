"""FastAPI 서버 통합 테스트.

httpx.AsyncClient + pytest.mark.anyio 사용.
CharacterOS 대신 _get_cos / _run_in_worker를 패칭하여 실측 의존성을 제거한다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import app

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Mock CharacterOS — 엔드포인트가 접근하는 속성만 제공
# ---------------------------------------------------------------------------


class _MockEmotion:
    def get_state(self) -> dict:
        return {"기쁨": 0.8, "슬픔": 0.1}


class _MockMemory:
    def snapshot_count(self) -> int:
        return 42


class _MockTurn:
    def __init__(self, role: str, content: str, timestamp: str) -> None:
        self.role = role
        self.content = content
        self.timestamp = timestamp


class _MockHistory:
    _turns = [
        _MockTurn("user", "안녕하세요", "2025-01-01T00:00:00"),
        _MockTurn("assistant", "안녕하세요! 반갑습니다.", "2025-01-01T00:00:01"),
    ]


class MockCharacterOS:
    """테스트용 CharacterOS 더블."""

    def __init__(self) -> None:
        self.emotion = _MockEmotion()
        self.memory = _MockMemory()
        self.history = _MockHistory()

    def chat(self, message: str) -> str:
        return f"Echo: {message}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_cos() -> MockCharacterOS:
    return MockCharacterOS()


@pytest.fixture()
async def client(mock_cos: MockCharacterOS):
    """패칭된 httpx AsyncClient (app lifespan 비활성)."""
    mock_run = AsyncMock(side_effect=lambda fn: fn())

    with (
        patch("src.api.server._get_cos", return_value=mock_cos),
        patch("src.api.server._run_in_worker", mock_run),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ---------------------------------------------------------------------------
# 1. GET /api/health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_returns_ok_status(self, client: AsyncClient):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 2. POST /api/chat
# ---------------------------------------------------------------------------


class TestChat:
    async def test_returns_response(self, client: AsyncClient):
        resp = await client.post("/api/chat", json={"message": "안녕"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Echo: 안녕"
        assert data["emotion"] == {"기쁨": 0.8, "슬픔": 0.1}

    async def test_missing_message_returns_422(self, client: AsyncClient):
        resp = await client.post("/api/chat", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3. GET /api/emotion
# ---------------------------------------------------------------------------


class TestEmotion:
    async def test_returns_emotion_state(self, client: AsyncClient):
        resp = await client.get("/api/emotion")
        assert resp.status_code == 200
        data = resp.json()
        assert data["기쁨"] == pytest.approx(0.8)
        assert data["슬픔"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 4. GET /api/memory/stats
# ---------------------------------------------------------------------------


class TestMemoryStats:
    async def test_returns_count(self, client: AsyncClient):
        resp = await client.get("/api/memory/stats")
        assert resp.status_code == 200
        assert resp.json() == {"count": 42}


# ---------------------------------------------------------------------------
# 5. GET /api/history
# ---------------------------------------------------------------------------


class TestHistory:
    async def test_returns_turns(self, client: AsyncClient):
        resp = await client.get("/api/history")
        assert resp.status_code == 200
        turns = resp.json()["turns"]
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "안녕하세요"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "안녕하세요! 반갑습니다."
