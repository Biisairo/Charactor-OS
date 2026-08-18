"""공통 테스트 fixtures — MockClient, 임시 경로, 테스트 캐릭터."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.agent.tools import FINISH_TOOL
from src.call_log import CallLogger
from src.character_layout import CharacterLayout
from src.character_os import CharacterOS
from src.llm.client import ToolCallPart, TrimmedMessage

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
        max_tokens: int | None = None,
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


# ---------------------------------------------------------------------------
# 뇌(ReAct) 테스트 지원 — 도구 호출 대본
#
# 뇌는 tool_calls로 말한다. MockClient는 텍스트만 돌려주므로,
# "몇 번째 호출에 어떤 도구를 부를지"를 대본으로 고정하는 더블이 따로 필요하다.
# ---------------------------------------------------------------------------


def tool_call(tool_name, /, *, raw_arguments: str | None = None, **arguments) -> ToolCallPart:
    """도구 호출 1건. `raw_arguments`를 주면 JSON 파싱 실패까지 흉내낼 수 있다."""
    args = raw_arguments if raw_arguments is not None else json.dumps(arguments, ensure_ascii=False)
    return ToolCallPart(id=f"call_{tool_name}", name=tool_name, arguments=args)


def finish_call(**payload) -> ToolCallPart:
    """종료 선언. 인자가 곧 응답 전략이다."""
    return tool_call(FINISH_TOOL, **payload)


def tool_step(*calls: ToolCallPart) -> TrimmedMessage:
    """도구를 호출하는 LLM 응답 1회."""
    return TrimmedMessage(
        content="", role="assistant", reasoning_content="", tool_calls=list(calls), usage=None
    )


def text_step(content: str) -> TrimmedMessage:
    """도구 없이 텍스트만 돌려주는 LLM 응답 1회."""
    return TrimmedMessage(
        content=content, role="assistant", reasoning_content="", tool_calls=[], usage=None
    )


class ScriptedClient:
    """대본대로 응답하는 더블. 예외를 넣으면 그 차례에 던진다.

    messages는 깊은 복사로 보관한다 — 뇌는 하나의 리스트를 계속 확장하므로,
    참조를 그대로 들고 있으면 모든 호출이 최종 상태로 보인다.
    """

    def __init__(self, steps: list):
        self._steps = list(steps)
        self.calls: list[dict] = []
        self.call_count = 0
        self.env = type("Env", (), {"model": "scripted-model"})()

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
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": tools or [],
                "max_tokens": max_tokens,
            }
        )
        self.call_count += 1

        if not self._steps:
            raise AssertionError("대본이 소진되었는데 호출이 더 들어왔다")

        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


# 파이프라인 테스트의 기본 뇌 동작 — 도구 셋을 한 번 훑고 종료한다.
# 뇌를 도입하기 전 고정 수집이 모으던 것과 같은 재료가 프롬프트에 오르도록 맞췄다.
DEFAULT_BRAIN_STRATEGY = {
    "situation": "사용자가 말을 걸었다",
    "intent": "성실하게 응답한다",
    "avoid": "설정에 없는 사실 지어내기",
    "tone": "평소 말투",
}


def default_brain_script() -> list[TrimmedMessage]:
    return [
        tool_step(
            tool_call("search_memory", query="사용자"),
            tool_call("get_history", n=10),
        ),
        tool_step(finish_call(**DEFAULT_BRAIN_STRATEGY)),
    ]


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
        brain_script: list | None = None,
    ):
        super().__init__(response=response)
        self.all_call_records: list[dict] = []
        self._fail_when = fail_when
        # 뇌 호출은 턴마다 다시 시작하므로 대본은 원본을 보관하고 턴 단위로 복제한다.
        self._brain_script_source = brain_script
        self._brain_steps: list = []
        self.brain_call_count = 0

    @staticmethod
    def _is_brain_call(tools) -> bool:
        """뇌 호출인가. 종료 도구가 목록에 있으면 뇌다."""
        return bool(tools) and any(t.get("function", {}).get("name") == FINISH_TOOL for t in tools)

    def _next_brain_step(self) -> TrimmedMessage:
        if not self._brain_steps:
            self._brain_steps = list(
                self._brain_script_source
                if self._brain_script_source is not None
                else default_brain_script()
            )
        step = self._brain_steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

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
        joined = " ".join(str(m.get("content", "")) for m in messages)
        if self._fail_when is not None and self._fail_when(joined):
            raise RuntimeError("주입된 LLM 실패")

        if self._is_brain_call(tools):
            self.brain_call_count += 1
            self.call_count += 1
            self.last_messages = messages
            self.all_call_records.append(
                {"messages": messages, "use_stream": use_stream, "response_format": response_format}
            )
            return self._next_brain_step()

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
        "working_memory_path": str(state_dir / "working_memory.json"),
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
