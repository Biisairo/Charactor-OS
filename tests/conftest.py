"""공통 테스트 fixtures — MockClient, 임시 경로, 테스트 캐릭터."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.llm.client import TrimmedMessage

# ---------------------------------------------------------------------------
# MockClient — 실제 LLM 호출 없이 고정 응답 반환
# ---------------------------------------------------------------------------


@dataclass
class MockResponse:
    """테스트에서 설정할 수 있는 Mock 응답."""

    content: str = "안녕하세요! 저는 홍길동입니다."
    reasoning_content: str = ""


class MockClient:
    """실제 LLM 호출을 하지 않는 더블.

    사용법:
        client = MockClient()
        client.next_response = MockResponse(content="원하는 응답")
        result = client.call_llm(messages=[...], tools=[], use_stream=False, mute=True)
    """

    def __init__(self, response: str = "안녕하세요! 저는 홍길동입니다."):
        self._response = MockResponse(content=response)
        self.call_count = 0
        self.last_messages = None
        self.last_kwargs = None

        # env 호환 (client.env.model 참조)
        self.env = type("Env", (), {"model": "mock-model"})()

    @property
    def next_response(self) -> MockResponse:
        return self._response

    @next_response.setter
    def next_response(self, value: MockResponse) -> None:
        self._response = value

    def call_llm(
        self,
        messages: list,
        tools: list,
        use_stream: bool,
        mute: bool,
        response_format: dict | None = None,
        token_callback=None,
    ) -> TrimmedMessage:
        self.call_count += 1
        self.last_messages = messages
        self.last_kwargs = {
            "tools": tools,
            "use_stream": use_stream,
            "mute": mute,
            "response_format": response_format,
        }

        content = self._response.content
        if token_callback and use_stream:
            for char in content:
                token_callback(char)

        return TrimmedMessage(
            content=content,
            role="assistant",
            reasoning_content=self._response.reasoning_content,
            tool_calls=[],
            usage=None,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MockClient:
    """기본 MockClient 인스턴스."""
    return MockClient()


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """임시 디렉토리 (테스트 종료 시 자동 삭제)."""
    return tmp_path


@pytest.fixture
def character_dir(tmp_path: Path) -> Path:
    """테스트용 캐릭터 디렉토리 (hong-gil-dong 복사)."""
    src = Path("characters/hong-gil-dong")
    dst = tmp_path / "characters" / "hong-gil-dong"
    if src.exists():
        shutil.copytree(src, dst)
    return dst


@pytest.fixture
def persona_path(character_dir: Path) -> str:
    """테스트용 페르소나 YAML 경로."""
    return str(character_dir / "persona.yaml")


@pytest.fixture
def examples_dir(character_dir: Path) -> str:
    """테스트용 few-shot 예시 디렉토리."""
    return str(character_dir / "examples")


@pytest.fixture
def knowledge_dir(character_dir: Path) -> str:
    """테스트용 지식 디렉토리."""
    return str(character_dir / "knowledge")
