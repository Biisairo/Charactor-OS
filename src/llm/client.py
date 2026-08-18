import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import openai
from dotenv import load_dotenv
from openai import (
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    Stream,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

MAX_RETRIES = 3

# 요청 타임아웃. 지정하지 않으면 openai SDK 기본값 600초가 적용되어,
# 느린 프로바이더 응답 하나가 워커 큐 전체를 10분간 막을 수 있다.
# 실측에서 374초 만에 **성공한** 호출이 있었다 (SPEC-09 §6.8).
DEFAULT_TIMEOUT_SECONDS = 120.0

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class LLMEnv:
    api_key: str
    model: str
    base_url: str


@dataclass
class ToolCallPart:
    id: str
    name: str
    arguments: str


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0


@dataclass
class TrimmedMessage:
    content: str
    role: str
    reasoning_content: str
    tool_calls: list[ToolCallPart]
    usage: TokenUsage | None = None


class Client:
    env: LLMEnv
    llm: openai.OpenAI

    def __init__(self, env: LLMEnv | None = None, timeout: float | None = None):
        """
        Args:
            env: LLM 접속 설정. 생략하면 환경 변수에서 읽는다.
                평가 판정자처럼 대화용과 다른 자격 증명·모델을 쓰는 경우에 주입한다.
            timeout: 요청 타임아웃(초). 생략하면 `OPENAI_TIMEOUT` 환경 변수,
                그것도 없으면 `DEFAULT_TIMEOUT_SECONDS`를 쓴다.
        """
        self.env = env or self._get_env()
        self.llm = openai.OpenAI(
            api_key=self.env.api_key,
            base_url=self.env.base_url,
            timeout=timeout if timeout is not None else self._get_timeout(),
        )

    @staticmethod
    def _get_timeout() -> float:
        load_dotenv()
        raw = os.getenv("OPENAI_TIMEOUT")
        if not raw:
            return DEFAULT_TIMEOUT_SECONDS
        return float(raw)

    # ---------------------------------------------------------------------------
    # Env
    # ---------------------------------------------------------------------------

    def _get_env(self) -> LLMEnv:
        load_dotenv()

        return LLMEnv(
            api_key=os.getenv("OPENAI_API_KEY") or "",
            model=os.getenv("OPENAI_MODEL") or "",
            base_url=os.getenv("OPENAI_BASE_URL") or "",
        )

    # ---------------------------------------------------------------------------
    # Core: LLM 호출 (stream / non-stream 분기만 다름)
    # ---------------------------------------------------------------------------

    def call_llm(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam],
        use_stream: bool,
        mute: bool,
        response_format: dict | None = None,
        token_callback: "Callable[[str], None] | None" = None,
        max_tokens: int | None = None,
    ) -> TrimmedMessage:
        """
        Args:
            max_tokens: 출력 상한. 프롬프트로 "간결하게"를 지시해도 모델은
                지키지 않을 수 있고, 생성 후 자르기는 지연을 줄이지 못한다.
                폭주를 구조적으로 막아야 하는 호출에만 지정한다.
        """
        for attempt in range(MAX_RETRIES):
            try:
                if use_stream:
                    return self._stream_to_message(
                        self.llm.chat.completions.create(
                            model=self.env.model,
                            messages=messages,
                            tools=tools,
                            stream=True,
                            response_format=response_format,
                            max_completion_tokens=max_tokens,
                            stream_options={"include_usage": True},
                        ),
                        mute,
                        token_callback,
                    )
                else:
                    return self._response_to_message(
                        self.llm.chat.completions.create(
                            model=self.env.model,
                            messages=messages,
                            tools=tools,
                            stream=False,
                            response_format=response_format,
                            max_completion_tokens=max_tokens,
                        ),
                        mute,
                    )
            except (RateLimitError, InternalServerError, APITimeoutError):
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2**attempt)
            except AuthenticationError:
                raise
            except BadRequestError:
                raise
        raise RuntimeError("unreachable")

    # ---------------------------------------------------------------------------
    # API: non-streaming -> TrimmedMessage
    # ---------------------------------------------------------------------------

    def _response_to_message(self, response: ChatCompletion, mute: bool) -> TrimmedMessage:
        msg = response.choices[0].message
        content = msg.content or ""
        reasoning_content = getattr(msg, "reasoning_content", "") or ""
        if not mute:
            print(reasoning_content, flush=True)
            print(content, flush=True)
        usage = None
        if response.usage:
            reasoning = (
                getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0) or 0
            )
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                reasoning_tokens=reasoning,
            )

        return TrimmedMessage(
            content=content,
            role=msg.role or "assistant",
            reasoning_content=reasoning_content,
            tool_calls=[
                ToolCallPart(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in (msg.tool_calls or [])
            ],
            usage=usage,
        )

    # ---------------------------------------------------------------------------
    # API: streaming -> TrimmedMessage
    # ---------------------------------------------------------------------------

    def _stream_to_message(
        self,
        stream: Stream[ChatCompletionChunk],
        mute: bool,
        token_callback: Callable[[str], None] | None = None,
    ) -> TrimmedMessage:
        content = ""
        reasoning_content = ""
        role = ""
        tool_calls_acc: dict[int, dict[str, str]] = {}
        usage: TokenUsage | None = None

        for chunk in stream:
            # 스트리밍 마지막 청크에 usage가 옴
            if chunk.usage:
                reasoning = (
                    getattr(chunk.usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                )
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    reasoning_tokens=reasoning,
                )
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                text = delta.content
                if not mute:
                    print(text, end="", flush=True)
                if token_callback:
                    token_callback(text)
                content += text

            rc = getattr(delta, "reasoning_content", None)
            if rc:
                reasoning_content += rc

            if delta.role:
                role = delta.role

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = ToolCallPart(id="", name="", arguments="")

                    if tc_delta.id is not None:
                        tool_calls_acc[idx].id = tc_delta.id
                    if tc_delta.function and tc_delta.function.name is not None:
                        tool_calls_acc[idx].name = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments is not None:
                        tool_calls_acc[idx].arguments += tc_delta.function.arguments

        if not mute:
            print("", flush=True)

        # dict[int, ...] -> list (index 순서 보장)
        tool_calls: list[ToolCallPart] = [tool_calls_acc[k] for k in sorted(tool_calls_acc)]

        return TrimmedMessage(
            content=content,
            role=role,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=usage,
        )
