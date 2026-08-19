"""토큰 계측 (SPEC-11).

프롬프트 예산은 자가 정확해야 의미가 있다. 종전 휴리스틱(`한글 1.5 / 기타 0.3`)은
실제 `prompt_tokens`를 중앙 +50.2% 과대 계상했고, 그 탓에 소민찌는 고정 섹션만으로
예산을 넘긴 것으로 계산되어 뇌가 모아온 검색 결과 4종을 통째로 버렸다 (P-1 · P-2).

실제 토크나이저로 센다. 다만 모델 공식 저장소는 `trust_remote_code`를 요구하므로
쓰지 않는다 — 토큰을 세자고 남의 코드를 실행할 이유가 없다 (P-6). 같은 계열의
토크나이저면 실측 오차 0.6%로 충분하다 (P-5).

**폴백은 침묵하지 않는다.** 토크나이저를 구할 수 없으면 휴리스틱으로 떨어지되
그 사실을 반드시 남긴다. 틀린 자로 재면서 그것을 모르는 것이 이 과제의
출발점이었다 (결정 2).

이 모듈에는 LLM 호출이 없다.
"""

from __future__ import annotations

from collections.abc import Callable

# 폴백 휴리스틱 계수.
#
# `response` 호출 535건에 최소제곱으로 맞춘 값이다. 종전 `1.5 / 0.3`은 평균절대
# 오차 49.8%였고 이 값은 1.2%다. 폴백이 쓰이는 상황일수록 정확해야 한다.
#
# 모델을 바꾸면 다시 틀어진다. 재보정은 `logs/llm_calls.jsonl`의 `response`
# 호출에서 (한글 자수, 기타 자수) → `prompt_tokens` 로 회귀하면 된다 (REQ-11-13).
HEURISTIC_KOREAN = 0.766
HEURISTIC_OTHER = 0.634

_HEURISTIC = "heuristic"

# 토크나이저 id → (토크나이저 또는 None, 폴백 사유).
#
# 프로세스 단위로 공유한다. 평가 하네스는 사례마다 `CharacterOS`를 새로 만들고,
# 인스턴스마다 다시 불러오면 1.4초 × 사례 수가 붙는다 (P-7). 실패도 캐시한다 —
# 안 그러면 네트워크가 없을 때 사례마다 다시 시도한다.
_CACHE: dict[str, tuple[object | None, str]] = {}


def clear_cache() -> None:
    """캐시를 비운다. 테스트가 로드 횟수를 세기 위한 진입점이다."""
    _CACHE.clear()


def _load_tokenizer(name: str):
    """Hugging Face 토크나이저를 불러온다.

    `trust_remote_code`를 넘기지 않는다 — 기본값이 False이며, 원격 코드를
    요구하는 저장소는 여기서 실패해 휴리스틱으로 떨어지는 것이 맞다.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


class TokenCounter:
    """문자열의 토큰 수를 센다.

    토크나이저 로드는 1.4초, 계측은 4KB에 1.5ms다. 로드를 한 번만 하면
    지연은 40~70초 턴에 묻힌다 (P-7).
    """

    def __init__(
        self,
        tokenizer_id: str = "",
        loader: Callable[[str], object] = _load_tokenizer,
    ):
        """
        Args:
            tokenizer_id: 쓸 토크나이저. 비우면 처음부터 휴리스틱으로 센다 —
                이는 폴백이 아니라 선택이므로 `fallback_reason`이 비어 있다.
            loader: 토크나이저를 만드는 함수. 테스트가 네트워크를 타지 않도록
                주입 지점을 연다.
        """
        self._tokenizer_id = tokenizer_id
        self._loader = loader
        self._tokenizer: object | None = None
        self._resolved = not tokenizer_id
        self._fallback_reason = ""

    # ─── 관측 (REQ-11-4) ───

    @property
    def method(self) -> str:
        """어느 자로 재고 있는가. `tokenizer:<id>` 또는 `heuristic`."""
        self._resolve()
        return f"tokenizer:{self._tokenizer_id}" if self._tokenizer else _HEURISTIC

    @property
    def fallback_reason(self) -> str:
        """토크나이저를 걸었으나 쓰지 못한 사유. 아니면 빈 문자열."""
        self._resolve()
        return self._fallback_reason

    # ─── 계측 ───

    def count(self, text: str) -> int:
        if not text:
            return 0

        self._resolve()
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))

        korean = sum(1 for char in text if "가" <= char <= "힣")
        return int(korean * HEURISTIC_KOREAN + (len(text) - korean) * HEURISTIC_OTHER)

    def _resolve(self) -> None:
        """토크나이저를 한 번만 불러온다. 실패는 폴백 사유로 남는다.

        결과는 프로세스 단위로 공유한다 — 성공도 실패도 한 번만 겪는다.
        """
        if self._resolved:
            return
        self._resolved = True

        cached = _CACHE.get(self._tokenizer_id)
        if cached is None:
            try:
                cached = (self._loader(self._tokenizer_id), "")
            except Exception as e:
                cached = (None, f"토크나이저 '{self._tokenizer_id}'를 쓸 수 없음 — {e}")
            _CACHE[self._tokenizer_id] = cached

        self._tokenizer, self._fallback_reason = cached


def from_config(config: dict, loader: Callable[[str], object] = _load_tokenizer) -> TokenCounter:
    """`config.yaml`의 `prompt` 섹션으로 계측기를 만든다 (REQ-11-6).

    섹션이 없으면 휴리스틱 계측기를 만든다 — 설정하지 않은 실행이 네트워크를
    타지 않게 하려는 것이다 (결정 4).
    """
    section = config.get("prompt") or {}
    return TokenCounter(tokenizer_id=str(section.get("tokenizer") or ""), loader=loader)
