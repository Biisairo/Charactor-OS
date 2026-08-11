"""
로컬 LLM 클라이언트 (mlx-lm)

기존 Client(OpenAI API)와 동일한 인터페이스로 로컬 모델을 호출한다.
- call_llm() → TrimmedMessage 반환
- 스트리밍 지원
- 모델 자동 로딩 & 캐싱
"""

from collections.abc import Callable
from dataclasses import dataclass

from mlx_lm import generate, load, stream_generate

# ---------------------------------------------------------------------------
# Types (기존 client.py와 동일)
# ---------------------------------------------------------------------------


@dataclass
class ToolCallPart:
    id: str
    name: str
    arguments: str


@dataclass
class TrimmedMessage:
    content: str
    role: str
    reasoning_content: str
    tool_calls: list[ToolCallPart]


# ---------------------------------------------------------------------------
# LocalClient
# ---------------------------------------------------------------------------


class LocalClient:
    """mlx-lm 기반 로컬 LLM 클라이언트"""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3.5-4B-MLX-4bit",
        adapter_path: str | None = None,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    # ---------------------------------------------------------------------------
    # 모델 로딩
    # ---------------------------------------------------------------------------

    def _ensure_loaded(self):
        """모델이 로드되지 않았으면 로드한다 (지연 로딩)"""
        if self._model is None:
            label = (
                f"{self.model_name} + LoRA({self.adapter_path})"
                if self.adapter_path
                else self.model_name
            )
            print(f"[LocalClient] 모델 로딩 중: {label}")
            self._model, self._tokenizer = load(
                self.model_name,
                adapter_path=self.adapter_path,
            )
            print("[LocalClient] 로딩 완료!")

    # ---------------------------------------------------------------------------
    # Core: LLM 호출
    # ---------------------------------------------------------------------------

    def call_llm(
        self,
        messages: list[dict[str, str]],
        tools: list | None = None,
        use_stream: bool = True,
        mute: bool = False,
        response_format: dict | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        token_callback: "Callable[[str], None] | None" = None,
    ) -> TrimmedMessage:
        """
        로컬 모델을 호출한다.

        Args:
            messages: 대화 메시지 리스트 [{"role": "user", "content": "..."}]
            tools: 도구 정의 (tool calling 지원 모델에서 사용)
            use_stream: 스트리밍 여부
            mute: 출력 억제 여부
            response_format: 응답 형식
            max_tokens: 최대 생성 토큰 수
            temperature: 생성 온도

        Returns:
            TrimmedMessage
        """
        self._ensure_loaded()

        # messages → 프롬프트 변환 (thinking mode 비활성화)
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        if use_stream:
            return self._stream_generate(prompt, max_tokens, temperature, mute, token_callback)
        else:
            return self._generate(prompt, max_tokens, temperature, mute, token_callback)

    # ---------------------------------------------------------------------------
    # 비스트리밍 생성
    # ---------------------------------------------------------------------------

    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        mute: bool,
        token_callback: Callable[[str], None] | None = None,
    ) -> TrimmedMessage:
        """비스트리밍 생성"""
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        content = self._extract_content(response)
        reasoning = self._extract_reasoning(response)

        if not mute:
            if reasoning:
                print(f"[Thinking] {reasoning}", flush=True)
            print(content, flush=True)

        if token_callback:
            token_callback(content)

        return TrimmedMessage(
            content=content,
            role="assistant",
            reasoning_content=reasoning,
            tool_calls=[],
        )

    # ---------------------------------------------------------------------------
    # 스트리밍 생성
    # ---------------------------------------------------------------------------

    def _stream_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        mute: bool,
        token_callback: Callable[[str], None] | None = None,
    ) -> TrimmedMessage:
        """스트리밍 생성"""
        full_text = ""
        in_think = False

        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        ):
            token = response.text if hasattr(response, "text") else str(response)
            full_text += token

            # thinking mode 처리
            if "<think>" in token and not in_think:
                in_think = True
                if not mute:
                    print("\n[Thinking] ", end="", flush=True)
                token = token.split("<think>")[-1]

            if "</think>" in token and in_think:
                in_think = False
                token = token.split("</think>")[0]

            if not mute:
                print(token, end="", flush=True)
            if token_callback and not in_think:
                token_callback(token)

        if not mute:
            print("", flush=True)

        content = self._extract_content(full_text)
        reasoning = self._extract_reasoning(full_text)

        return TrimmedMessage(
            content=content,
            role="assistant",
            reasoning_content=reasoning,
            tool_calls=[],
        )

    # ---------------------------------------------------------------------------
    # Thinking mode 파싱
    # ---------------------------------------------------------------------------

    def _extract_content(self, text: str) -> str:
        """thinking mode를 제외한 실제 응답 추출"""
        # Qwen3.5 형식: "Thinking Process:\n...\n\n" 다음이 실제 응답
        if "Thinking Process:" in text:
            parts = text.split("Thinking Process:", 1)
            if len(parts) > 1:
                # thinking 부분 이후의 첫 번째 줄바꿈 이후가 실제 응답
                think_and_content = parts[1]
                # 연속된 줄바꿈으로 thinking과 content 분리
                lines = think_and_content.split("\n")
                content_start = 0
                found_empty = False
                for i, line in enumerate(lines):
                    if line.strip() == "":
                        if found_empty:
                            content_start = i + 1
                            break
                        found_empty = True
                    else:
                        found_empty = False
                return "\n".join(lines[content_start:]).strip()

        # <think> 형식도 지원
        if "<think>" in text and "</think>" in text:
            start = text.find("</think>") + len("</think>")
            return text[start:].strip()

        return text.strip()

    def _extract_reasoning(self, text: str) -> str:
        """thinking mode의 reasoning 추출"""
        # Qwen3.5 형식
        if "Thinking Process:" in text:
            parts = text.split("Thinking Process:", 1)
            if len(parts) > 1:
                think_and_content = parts[1]
                lines = think_and_content.split("\n")
                think_lines = []
                found_empty = False
                for line in lines:
                    if line.strip() == "":
                        if found_empty:
                            break
                        found_empty = True
                    else:
                        found_empty = False
                        think_lines.append(line)
                return "\n".join(think_lines).strip()

        # <think> 형식
        if "<think>" in text and "</think>" in text:
            start = text.find("<think>") + len("<think>")
            end = text.find("</think>")
            return text[start:end].strip()

        return ""
