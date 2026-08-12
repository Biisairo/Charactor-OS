"""공통 테스트 fixtures — MockClient, 임시 경로, 테스트 캐릭터."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.call_log import CallLogger
from src.character_layout import CharacterLayout
from src.character_os import CharacterOS
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
    return str(CharacterLayout.of(character_dir).persona_path)


@pytest.fixture
def examples_dir(character_dir: Path) -> str:
    """테스트용 few-shot 예시 디렉토리."""
    return str(CharacterLayout.of(character_dir).examples_dir)


@pytest.fixture
def knowledge_dir(character_dir: Path) -> str:
    """테스트용 지식 디렉토리."""
    return str(CharacterLayout.of(character_dir).knowledge_dir)


# ---------------------------------------------------------------------------
# 파이프라인 테스트 지원 — 임베딩 우회, 선택적 실패 클라이언트, CharacterOS 조립
#
# 실제 LLM과 sentence-transformers를 모두 배제하여, 파이프라인 테스트가
# API 키 없이 결정론적으로 동작하도록 한다 (REQ-02-9).
# ---------------------------------------------------------------------------

# 감정·기억 갱신은 각각 별도 LLM 호출이다. 프롬프트 본문의 고유 문구로
# 어느 호출인지 식별하여, 특정 단계만 실패시킬 수 있게 한다.
EMOTION_PROMPT_MARKER = "감정 상태를 업데이트하세요"
MEMORY_PROMPT_MARKER = "구체적인 사실**만 추출하세요"


def deterministic_embed(text: str) -> np.ndarray:
    """텍스트에 대해 항상 같은 벡터를 반환하는 더미 임베딩.

    sentence-transformers 모델 로드(수백 MB)를 건너뛰기 위한 대체물이다.
    """
    rng = np.random.RandomState(hash(text) % (2**31))
    vec = rng.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture(autouse=True)
def patch_embedding(monkeypatch):
    """모든 테스트에서 SentenceTransformer 로드를 우회한다."""
    stub = type(
        "EmbeddingStub",
        (),
        {"encode": staticmethod(lambda text, normalize_embeddings=True: deterministic_embed(text))},
    )()
    monkeypatch.setattr(CharacterOS, "_embedding_model", stub)


class PipelineMockClient(MockClient):
    """파이프라인 전 구간에서 쓰이는 MockClient.

    `CharacterOS._generate_response`는 call_llm(messages, use_stream, mute)로
    호출하지만 emotion/memory.update는 response_format을 추가로 전달한다.
    tools를 선택적으로 만들어 두 경로를 모두 받는다.

    fail_when을 주면 해당 프롬프트에 대해서만 예외를 던진다 — 후처리 중
    특정 단계만 실패시켜 롤백을 검증하기 위한 장치다.
    """

    def __init__(
        self,
        response: str = "안녕하세요! 저는 홍길동입니다.",
        fail_when: Callable[[str], bool] | None = None,
    ):
        super().__init__(response=response)
        self.all_call_records: list[dict] = []
        self._fail_when = fail_when

    def call_llm(
        self,
        messages,
        tools=None,
        use_stream=False,
        mute=True,
        response_format=None,
        token_callback=None,
    ) -> TrimmedMessage:
        joined = " ".join(str(m.get("content", "")) for m in messages)
        if self._fail_when is not None and self._fail_when(joined):
            raise RuntimeError("주입된 LLM 실패")

        self.all_call_records.append(
            {
                "messages": messages,
                "use_stream": use_stream,
                "response_format": response_format,
            }
        )
        return super().call_llm(
            messages=messages,
            tools=tools or [],
            use_stream=use_stream,
            mute=mute,
            response_format=response_format,
            token_callback=token_callback,
        )


def make_character_os(
    character_dir: Path,
    state_dir: Path,
    client: MockClient,
    **overrides,
) -> CharacterOS:
    """상태 파일을 `state_dir`에 격리한 CharacterOS를 만든다."""
    kwargs = {
        "character_dir": str(character_dir),
        "memory_db_path": str(state_dir / "memories.db"),
        "emotion_save_path": str(state_dir / "emotions.json"),
        "history_save_path": str(state_dir / "history.json"),
        "debug": False,
        "output": lambda _msg: None,
        "model_type": "api",
        "no_review": True,
        "client": client,
        # 테스트가 실제 운영 로그 파일에 쓰지 않도록 비활성 로거를 주입한다
        "call_logger": CallLogger(enabled=False),
    }
    kwargs.update(overrides)
    return CharacterOS(**kwargs)
