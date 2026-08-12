"""프로바이더 오류 응답 판별 (TASK-01 후속).

이 판별이 없으면 API 거부 메시지가 '품질 1점'으로 집계되어 지표를 오염시킨다.
반대로 과하게 걸러내면 실제 품질 문제를 놓치므로, 두 방향 모두 검증한다.
"""

from __future__ import annotations

import pytest

from src.validity import provider_error_reason


class TestDetectsProviderErrors:
    def test_high_risk_rejection(self):
        """실제로 겪은 케이스 — reflection-off 3건이 이 응답을 받았다."""
        reason = provider_error_reason(
            "The request was rejected because it was considered high risk"
        )

        assert reason is not None
        assert "프로바이더 오류" in reason

    @pytest.mark.parametrize(
        "response",
        [
            "Rate limit exceeded, please retry",
            "Internal Server Error",
            "This violates our content policy.",
            "502 Bad Gateway",
        ],
    )
    def test_known_error_shapes(self, response: str):
        assert provider_error_reason(response) is not None

    def test_case_insensitive(self):
        assert provider_error_reason("THE REQUEST WAS REJECTED") is not None

    @pytest.mark.parametrize("response", ["", "   ", "\n\t "])
    def test_empty_response(self, response: str):
        assert provider_error_reason(response) == "빈 응답"


class TestKeepsRealResponses:
    def test_normal_korean_response(self):
        assert provider_error_reason("흠, 그리 말씀하시니 반갑소.") is None

    def test_wrong_language_is_quality_not_error(self):
        """중국어 응답은 실제 품질 결함이다. 채점에서 빼면 결함이 은폐된다."""
        assert provider_error_reason("何足挂齿。能助你一臂之力，我也欣慰。") is None

    def test_code_in_response_is_quality_not_error(self):
        """페르소나를 깨고 코드를 뱉은 것도 채점 대상이다."""
        response = "흠, 그러한 계산법이로구나.\n```python\ndef fib(n): return n\n```"

        assert provider_error_reason(response) is None

    def test_english_but_in_character(self):
        """영어가 섞였다는 이유만으로 제외해서는 안 된다."""
        assert provider_error_reason("흠, 그 'system'이라는 말은 무엇이오?") is None

    def test_long_response_unaffected(self):
        assert provider_error_reason("한양의 밤은 고요하오. " * 50) is None
