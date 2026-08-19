"""`config.yaml` 로드.

CLI·API·평가 하네스가 같은 설정을 읽어야 한다. 사본이 갈라지면 평가가
런타임과 다른 설정으로 측정하게 된다 (SPEC-11 REQ-11-11).

이 모듈에는 LLM 호출이 없다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = "config.yaml"


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """설정을 읽는다. 파일이 없으면 빈 설정이다 — 없는 것은 오류가 아니다."""
    config_file = Path(path)
    if not config_file.exists():
        return {}
    with open(config_file, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
