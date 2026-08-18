"""LLM 요청 타임아웃 (SPEC-09 §6.8 결함 1).

타임아웃이 없으면 openai SDK 기본값 600초가 적용된다. 실측에서 374초 만에
**성공한** 호출이 있었고, 웹 UI는 캐릭터당 스레드 하나로 직렬 처리하므로
그 6분 동안 이후 요청 전부가 큐에서 대기했다. 사용자에게는 멈춤으로 보인다.
"""

from __future__ import annotations

import openai
import pytest

from src.llm.client import DEFAULT_TIMEOUT_SECONDS, Client, LLMEnv


def _env() -> LLMEnv:
    return LLMEnv(api_key="k", model="m", base_url="http://localhost:9/v1")


class TestTimeoutIsConfigured:
    def test_default_timeout_is_applied(self):
        client = Client(env=_env())

        assert client.llm.timeout == DEFAULT_TIMEOUT_SECONDS

    def test_timeout_is_overridable(self):
        client = Client(env=_env(), timeout=42.0)

        assert client.llm.timeout == 42.0

    def test_default_is_well_under_sdk_default(self):
        """SDK 기본 600초를 그대로 쓰면 방어가 아무 의미가 없다."""
        assert DEFAULT_TIMEOUT_SECONDS < 600


class TestTimeoutIsRetried:
    def test_timeout_is_retried_then_raised(self, monkeypatch):
        client = Client(env=_env())
        attempts = {"n": 0}

        def _always_timeout(**kwargs):
            attempts["n"] += 1
            raise openai.APITimeoutError(request=None)

        monkeypatch.setattr(client.llm.chat.completions, "create", _always_timeout)
        monkeypatch.setattr("time.sleep", lambda _s: None)

        with pytest.raises(openai.APITimeoutError):
            client.call_llm(messages=[], tools=[], use_stream=False, mute=True)

        assert attempts["n"] > 1, "타임아웃은 일시적 지연일 수 있으므로 재시도해야 한다"
