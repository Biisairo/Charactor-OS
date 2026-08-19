"""토큰 계측 (SPEC-11).

예산을 틀린 자로 재고 있었다. `한글 1.5 / 기타 0.3` 계수가 실제 프롬프트
토큰을 중앙 +50.2% 과대 계상했고, 그 결과 소민찌는 검색 결과 4종을 통째로
버렸다 (SPEC-11 P-1 · P-2).

폴백은 침묵하지 않아야 한다. 틀린 자로 재면서 그것을 모르는 것이 이 과제의
출발점이다 (결정 2).

네트워크를 타지 않는다 — 실제 토크나이저는 더블로 주입한다.
"""

from __future__ import annotations

import pytest

from src.prompts.tokens import (
    HEURISTIC_KOREAN,
    HEURISTIC_OTHER,
    TokenCounter,
    clear_cache,
    from_config,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    """토크나이저 캐시는 프로세스 단위로 공유된다. 테스트끼리 새지 않게 비운다."""
    clear_cache()
    yield
    clear_cache()


class _Tokenizer:
    """토큰 하나가 두 글자인 가짜 토크나이저."""

    def encode(self, text: str) -> list[int]:
        return [0] * ((len(text) + 1) // 2)


def _boom(name: str):
    raise OSError(f"내려받을 수 없음: {name}")


class TestHeuristicFallback:
    def test_missing_tokenizer_falls_back(self) -> None:
        """T-1: 예외를 던지지 않는다."""
        counter = TokenCounter(tokenizer_id="없는/모델", loader=_boom)

        assert counter.count("안녕하세요") > 0

    def test_fallback_is_reported(self) -> None:
        """T-2: 폴백은 침묵하지 않는다 (REQ-11-4)."""
        counter = TokenCounter(tokenizer_id="없는/모델", loader=_boom)
        counter.count("안녕하세요")

        assert counter.method == "heuristic"
        assert counter.fallback_reason
        assert "없는/모델" in counter.fallback_reason

    def test_no_tokenizer_configured_is_not_a_fallback(self) -> None:
        """토크나이저를 아예 안 걸었으면 폴백이 아니라 선택이다."""
        counter = TokenCounter()

        assert counter.method == "heuristic"
        assert counter.fallback_reason == ""

    def test_uses_calibrated_coefficients(self) -> None:
        """T-4 · REQ-11-5: 종전 계수(1.5/0.3)보다 작게 센다."""
        counter = TokenCounter()
        text = "안녕하세요 반갑습니다"
        korean = sum(1 for c in text if "가" <= c <= "힣")
        legacy = int(korean * 1.5 + (len(text) - korean) * 0.3)

        assert counter.count(text) == int(
            korean * HEURISTIC_KOREAN + (len(text) - korean) * HEURISTIC_OTHER
        )
        assert counter.count(text) < legacy

    def test_empty_text_is_zero(self) -> None:
        """T-5."""
        assert TokenCounter().count("") == 0


class TestRealTokenizer:
    def test_uses_injected_tokenizer(self) -> None:
        """T-3."""
        counter = TokenCounter(tokenizer_id="가짜/토크나이저", loader=lambda _n: _Tokenizer())

        assert counter.count("abcd") == 2
        assert counter.method == "tokenizer:가짜/토크나이저"
        assert counter.fallback_reason == ""

    def test_loader_is_called_once(self) -> None:
        """로드는 1.4초다. 턴마다 다시 만들면 안 된다 (P-7)."""
        calls: list[str] = []

        def loader(name: str):
            calls.append(name)
            return _Tokenizer()

        counter = TokenCounter(tokenizer_id="가짜/토크나이저", loader=loader)
        counter.count("가")
        counter.count("나")

        assert calls == ["가짜/토크나이저"]

    def test_never_trusts_remote_code(self) -> None:
        """REQ-11-2 · P-6: 토큰을 세자고 남의 코드를 실행하지 않는다.

        금지 대상은 낱말이 아니라 **넘기는 것**이다. 주석은 왜 안 넘기는지를
        설명해야 하므로 낱말 자체는 소스에 남는다.
        """
        import inspect
        import re

        from src.prompts import tokens

        source = inspect.getsource(tokens)

        assert not re.search(r"trust_remote_code\s*=\s*True", source)


class TestFromConfig:
    def test_absent_section_yields_heuristic(self) -> None:
        """T-6: 설정이 없으면 기존 동작을 유지한다 (REQ-11-6)."""
        counter = from_config({})

        assert counter.method == "heuristic"
        assert counter.fallback_reason == ""

    def test_tokenizer_from_config_is_attempted(self) -> None:
        """T-7."""
        seen: list[str] = []

        counter = from_config(
            {"prompt": {"tokenizer": "Qwen/Qwen2.5-7B-Instruct"}},
            loader=lambda n: seen.append(n) or _Tokenizer(),
        )
        counter.count("가나다라")

        assert seen == ["Qwen/Qwen2.5-7B-Instruct"]
        assert counter.method == "tokenizer:Qwen/Qwen2.5-7B-Instruct"


class TestTokenizerCache:
    """평가 하네스는 사례마다 `CharacterOS`를 새로 만든다.

    인스턴스마다 토크나이저를 다시 불러오면 1.4초 × 사례 수가 붙는다 (P-7).
    """

    def test_tokenizer_is_shared_across_instances(self) -> None:
        loads: list[str] = []

        def loader(name: str):
            loads.append(name)
            return _Tokenizer()

        for _ in range(3):
            TokenCounter(tokenizer_id="공유/토크나이저", loader=loader).count("가나")

        assert loads == ["공유/토크나이저"]

    def test_failure_is_not_retried_per_instance(self) -> None:
        """실패도 캐시한다 — 사례마다 네트워크를 다시 타면 안 된다."""
        attempts: list[str] = []

        def loader(name: str):
            attempts.append(name)
            raise OSError("네트워크 없음")

        for _ in range(3):
            counter = TokenCounter(tokenizer_id="실패/토크나이저", loader=loader)
            counter.count("가나")
            assert counter.method == "heuristic"

        assert len(attempts) == 1
