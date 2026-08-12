"""런타임 싱글톤과 그 접근자.

`CharacterOS`는 서버당 하나이며 전용 스레드 워커가 감싼다. 라우터가 이 모듈을
통해서만 런타임에 닿게 하여, 라우터끼리는 서로를 모르게 한다.

설정도 여기 둔다. `lifespan`이 워커를 만들 때 읽고, 캐릭터 전환처럼 런타임에
`CharacterOS`를 다시 만드는 곳에서도 같은 값을 쓴다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml

from src.api.worker import CharacterWorker
from src.character_os import CharacterOS

DEFAULT_CONFIG = {
    "character_dir": "characters/hong-gil-dong",
    "memory_db_path": "memory/memories.db",
    "emotion_save_path": "memory/emotions.json",
    "history_save_path": "memory/history.json",
    "model_type": "api",
    "local_model": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "adapter_path": None,
}


def load_config(path: str) -> dict:
    config_file = Path(path)
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# lifespan에서 초기화된다. 그 전에 접근하면 명시적으로 실패한다 —
# None을 돌려주면 호출부마다 None 검사가 흩어진다.
_worker: CharacterWorker | None = None
_config: dict = {}


def set_worker(worker: CharacterWorker | None) -> None:
    global _worker
    _worker = worker


def get_worker() -> CharacterWorker | None:
    return _worker


def set_config(config: dict) -> None:
    global _config
    _config = config


def get_config() -> dict:
    return _config


def get_cos() -> CharacterOS:
    if _worker is None:
        raise RuntimeError("CharacterOS not initialized")
    return _worker.cos


async def run_in_worker(fn: Callable) -> any:
    """CharacterWorker에서 함수를 실행하고 결과를 반환한다."""
    if _worker is None:
        raise RuntimeError("CharacterOS not initialized")
    return await _worker.run(fn)
