"""토큰 사용량 → 비용 추정.

단가는 코드가 아니라 `pricing.yaml`에서 읽는다. 가격은 프로바이더 사정으로
자주 바뀌므로 코드 변경 없이 갱신할 수 있어야 한다.

등록되지 않은 모델은 0이 아니라 `None`을 반환한다. 0으로 처리하면
"비용이 들지 않는다"로 잘못 읽힌다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PRICING_PATH = Path(__file__).parent.parent / "pricing.yaml"

TOKENS_PER_UNIT = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """1M 토큰당 USD 단가."""

    input: float
    output: float


def parse_price_table(payload: dict | None) -> dict[str, ModelPrice]:
    """설정 매핑에서 단가표를 만든다. 파일 접근이 없어 단위 테스트가 쉽다."""
    if not payload:
        return {}

    models = payload.get("models") or {}
    table: dict[str, ModelPrice] = {}
    for name, entry in models.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("input") is None or entry.get("output") is None:
            continue
        table[str(name)] = ModelPrice(
            input=float(entry["input"]),
            output=float(entry["output"]),
        )
    return table


def load_price_table(path: Path | None = None) -> dict[str, ModelPrice]:
    """단가표를 읽는다. 파일이 없으면 빈 표를 반환한다 (비용 미등록으로 표시됨)."""
    target = path or DEFAULT_PRICING_PATH
    if not target.exists():
        return {}
    return parse_price_table(yaml.safe_load(target.read_text(encoding="utf-8")))


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    table: dict[str, ModelPrice],
) -> float | None:
    """USD 추정 비용. 단가가 없는 모델이면 None."""
    price = table.get(model)
    if price is None:
        return None

    cost = (prompt_tokens * price.input + completion_tokens * price.output) / TOKENS_PER_UNIT
    return round(cost, 6)


def format_cost(cost: float | None) -> str:
    """사람이 읽는 비용 표기. 미등록 모델임을 감추지 않는다."""
    if cost is None:
        return "단가 미등록"
    return f"${cost:.6f}"
