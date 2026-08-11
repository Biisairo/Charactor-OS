"""평가 판정자 설정 (REQ-01-10, REQ-01-12).

판정자는 대화용과 완전히 분리된 자격 증명·엔드포인트·모델을 사용한다.
판정자가 평가 대상과 같은 모델이면 자기 편향이 생기기 때문이다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from src.llm.client import Client, LLMEnv

# 판정자 설정 환경 변수
ENV_API_KEY = "EVAL_API_KEY"
ENV_MODEL = "EVAL_MODEL"
ENV_BASE_URL = "EVAL_BASE_URL"

# 대화 대상 설정 환경 변수 (기존 대화 경로와 동일)
ENV_TARGET_API_KEY = "OPENAI_API_KEY"
ENV_TARGET_MODEL = "OPENAI_MODEL"


class EvalConfigError(RuntimeError):
    """평가 설정이 불완전할 때 발생한다."""


@dataclass(frozen=True)
class JudgeConfig:
    """판정자 접속 설정."""

    api_key: str
    model: str
    base_url: str

    def to_env(self) -> LLMEnv:
        return LLMEnv(api_key=self.api_key, model=self.model, base_url=self.base_url)

    def build_client(self) -> Client:
        return Client(env=self.to_env())


def missing_judge_vars(environ: dict[str, str] | None = None) -> list[str]:
    """판정에 필요한데 비어 있는 환경 변수 이름을 반환한다.

    순수 함수 — 실제 환경과 무관하게 테스트할 수 있도록 environ을 주입받는다.
    base_url은 비워두면 프로바이더 기본값을 쓰므로 필수가 아니다.
    """
    env = environ if environ is not None else dict(os.environ)
    return [name for name in (ENV_API_KEY, ENV_MODEL) if not (env.get(name) or "").strip()]


def load_judge_config(environ: dict[str, str] | None = None) -> JudgeConfig:
    """판정자 설정을 읽는다. 누락된 항목이 있으면 무엇이 없는지 명시하고 실패한다."""
    if environ is None:
        load_dotenv()
        environ = dict(os.environ)

    missing = missing_judge_vars(environ)
    if missing:
        raise EvalConfigError(
            "평가 설정이 없습니다: "
            + ", ".join(missing)
            + "\n.env에 다음을 지정하세요 (.env.example 참조):\n"
            + "\n".join(f"  {name}=..." for name in missing)
        )

    return JudgeConfig(
        api_key=environ[ENV_API_KEY].strip(),
        model=environ[ENV_MODEL].strip(),
        base_url=(environ.get(ENV_BASE_URL) or "").strip(),
    )


def target_model_name(environ: dict[str, str] | None = None) -> str:
    """평가 대상(대화용) 모델명. 결과 기록용이다 (REQ-01-11)."""
    if environ is None:
        load_dotenv()
        environ = dict(os.environ)
    return (environ.get(ENV_TARGET_MODEL) or "").strip() or "(미지정)"
