"""평가 판정자 설정 검증 (TASK-01 / REQ-01-10, REQ-01-12).

판정자 설정은 대화용과 분리되어야 하며, 누락 시 무엇이 없는지 명확히
알려주고 즉시 실패해야 한다. 조용히 대화용 설정으로 넘어가면 자기 편향이
생긴 채로 평가가 진행된다.
"""

from __future__ import annotations

import pytest

from eval.config import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    EvalConfigError,
    load_judge_config,
    missing_judge_vars,
    target_model_name,
)

FULL_ENV = {
    ENV_API_KEY: "sk-eval-key",
    ENV_MODEL: "gpt-4o",
    ENV_BASE_URL: "https://judge.example.com/v1",
}


class TestMissingVars:
    def test_none_missing_when_complete(self):
        assert missing_judge_vars(FULL_ENV) == []

    def test_base_url_is_optional(self):
        """엔드포인트를 비우면 프로바이더 기본값을 쓰므로 필수가 아니다."""
        assert missing_judge_vars({ENV_API_KEY: "k", ENV_MODEL: "m"}) == []

    def test_reports_all_missing(self):
        assert missing_judge_vars({}) == [ENV_API_KEY, ENV_MODEL]

    def test_blank_counts_as_missing(self):
        assert missing_judge_vars({ENV_API_KEY: "   ", ENV_MODEL: "m"}) == [ENV_API_KEY]


class TestLoadJudgeConfig:
    def test_loads_all_fields(self):
        config = load_judge_config(FULL_ENV)

        assert config.api_key == "sk-eval-key"
        assert config.model == "gpt-4o"
        assert config.base_url == "https://judge.example.com/v1"

    def test_strips_whitespace(self):
        config = load_judge_config({ENV_API_KEY: " k ", ENV_MODEL: " m "})

        assert config.api_key == "k"
        assert config.model == "m"

    def test_error_names_missing_vars(self):
        with pytest.raises(EvalConfigError) as exc:
            load_judge_config({})

        message = str(exc.value)
        assert ENV_API_KEY in message
        assert ENV_MODEL in message

    def test_error_mentions_only_what_is_missing(self):
        with pytest.raises(EvalConfigError) as exc:
            load_judge_config({ENV_MODEL: "gpt-4o"})

        message = str(exc.value)
        assert ENV_API_KEY in message
        assert f"{ENV_MODEL}=..." not in message

    def test_does_not_fall_back_to_conversation_key(self):
        """대화용 키가 있어도 판정자 키가 없으면 실패해야 한다."""
        with pytest.raises(EvalConfigError):
            load_judge_config({"OPENAI_API_KEY": "sk-target", "OPENAI_MODEL": "gpt-4o-mini"})

    def test_to_env_maps_fields(self):
        env = load_judge_config(FULL_ENV).to_env()

        assert env.api_key == "sk-eval-key"
        assert env.model == "gpt-4o"


class TestTargetModelName:
    def test_reads_conversation_model(self):
        assert target_model_name({"OPENAI_MODEL": "gpt-4o-mini"}) == "gpt-4o-mini"

    def test_placeholder_when_unset(self):
        assert target_model_name({}) == "(미지정)"
