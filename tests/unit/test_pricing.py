"""비용 추정 (TASK-04 / REQ-04-3, 04-6).

단가는 코드가 아니라 설정에서 온다. 단가를 바꾸면 비용이 그에 비례해
변해야 하고, 등록되지 않은 모델은 0이 아니라 '미등록'으로 드러나야 한다.
0으로 처리하면 비용이 없는 것처럼 읽힌다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.pricing import (
    DEFAULT_PRICING_PATH,
    ModelPrice,
    estimate_cost,
    format_cost,
    load_price_table,
    parse_price_table,
)

TABLE = {"m": ModelPrice(input=2.0, output=10.0)}


class TestParsePriceTable:
    def test_parses_models(self):
        table = parse_price_table({"models": {"a": {"input": 1.0, "output": 2.0}}})

        assert table["a"] == ModelPrice(input=1.0, output=2.0)

    def test_ignores_incomplete_entries(self):
        """단가가 반쪽만 있으면 잘못된 비용을 만드느니 등록하지 않는다."""
        table = parse_price_table(
            {"models": {"a": {"input": 1.0}, "b": {"output": 2.0}, "c": "문자열"}}
        )

        assert table == {}

    @pytest.mark.parametrize("payload", [None, {}, {"models": None}])
    def test_empty_payload(self, payload):
        assert parse_price_table(payload) == {}


class TestEstimateCost:
    def test_basic(self):
        # 1M 입력 × $2 + 1M 출력 × $10 = $12
        assert estimate_cost("m", 1_000_000, 1_000_000, TABLE) == 12.0

    def test_scales_with_tokens(self):
        assert estimate_cost("m", 500_000, 0, TABLE) == 1.0

    def test_input_and_output_priced_separately(self):
        """출력 토큰이 더 비싸다. 합산으로 뭉개면 비용을 과소평가한다."""
        input_only = estimate_cost("m", 1_000_000, 0, TABLE)
        output_only = estimate_cost("m", 0, 1_000_000, TABLE)

        assert output_only > input_only

    def test_unknown_model_returns_none(self):
        assert estimate_cost("모르는모델", 1000, 1000, TABLE) is None

    def test_empty_table_returns_none(self):
        assert estimate_cost("m", 1000, 1000, {}) is None

    def test_zero_tokens(self):
        assert estimate_cost("m", 0, 0, TABLE) == 0.0

    def test_price_change_scales_cost(self):
        """REQ-04-3 — 단가표를 바꾸면 비용이 비례해 변해야 한다."""
        base = estimate_cost("m", 1_000_000, 1_000_000, TABLE)
        doubled = estimate_cost(
            "m", 1_000_000, 1_000_000, {"m": ModelPrice(input=4.0, output=20.0)}
        )

        assert doubled == base * 2


class TestFormatCost:
    def test_known_cost(self):
        assert format_cost(0.001234) == "$0.001234"

    def test_unknown_is_not_zero(self):
        """미등록을 $0으로 표기하면 '공짜'로 오해된다."""
        assert format_cost(None) == "단가 미등록"
        assert "0" not in format_cost(None)


class TestLoadPriceTable:
    def test_missing_file_yields_empty(self, tmp_path: Path):
        assert load_price_table(tmp_path / "없음.yaml") == {}

    def test_reads_file(self, tmp_path: Path):
        path = tmp_path / "pricing.yaml"
        path.write_text(
            yaml.dump({"models": {"x": {"input": 3.0, "output": 6.0}}}), encoding="utf-8"
        )

        assert load_price_table(path)["x"].output == 6.0

    def test_shipped_file_is_valid(self):
        """리포지토리에 담긴 단가표가 깨져 있으면 안 된다."""
        table = load_price_table(DEFAULT_PRICING_PATH)

        assert table, "pricing.yaml에 최소 한 개 모델이 등록되어 있어야 한다"
        for name, price in table.items():
            assert price.input >= 0 and price.output >= 0, name
