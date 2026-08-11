"""HistoryModule 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path

from src.modules.history import HistoryModule

# ── helpers ──────────────────────────────────────────────────────────────────


def _make(path: Path | None = None, max_turns: int = 100) -> HistoryModule:
    return HistoryModule(save_path=str(path) if path else None, max_turns=max_turns)


def _fill(h: HistoryModule, n: int, *, prefix: str = "msg") -> None:
    for i in range(n):
        role = "user" if i % 2 == 0 else "character"
        h.add_turn(role, f"{prefix} {i}")


# ── add_turn / count ─────────────────────────────────────────────────────────


class TestAddTurn:
    def test_add_single_turn(self) -> None:
        h = _make()
        assert h.count() == 0
        h.add_turn("user", "hello")
        assert h.count() == 1

    def test_add_multiple_turns_increments_count(self) -> None:
        h = _make()
        _fill(h, 5)
        assert h.count() == 5

    def test_turn_fields(self) -> None:
        h = _make()
        h.add_turn("character", "안녕")
        turn = h.get_recent(1)[0]
        assert turn.role == "character"
        assert turn.content == "안녕"
        assert isinstance(turn.timestamp, float)


# ── get_recent ────────────────────────────────────────────────────────────────


class TestGetRecent:
    def test_returns_last_n(self) -> None:
        h = _make()
        _fill(h, 10)
        recent = h.get_recent(3)
        assert len(recent) == 3
        assert [t.content for t in recent] == ["msg 7", "msg 8", "msg 9"]

    def test_n_greater_than_count_returns_all(self) -> None:
        h = _make()
        _fill(h, 3)
        assert len(h.get_recent(100)) == 3

    def test_n_zero_returns_all(self) -> None:
        """Python list[-0:] == list[:] — returns all turns, not empty."""
        h = _make()
        _fill(h, 5)
        assert len(h.get_recent(0)) == 5

    def test_empty_history(self) -> None:
        h = _make()
        assert h.get_recent(5) == []


# ── to_prompt ────────────────────────────────────────────────────────────────


class TestToPrompt:
    def test_empty_history(self) -> None:
        h = _make()
        assert h.to_prompt() == "[최근 대화]\n대화 없음"

    def test_format_single_turn(self) -> None:
        h = _make()
        h.add_turn("user", "hi")
        assert h.to_prompt(1) == "[최근 대화]\n사용자: hi"

    def test_format_multiple_turns(self) -> None:
        h = _make()
        h.add_turn("user", "안녕")
        h.add_turn("character", "안녕하세요!")
        result = h.to_prompt(2)
        assert result == "[최근 대화]\n사용자: 안녕\n캐릭터: 안녕하세요!"

    def test_n_limits_output(self) -> None:
        h = _make()
        _fill(h, 5)
        result = h.to_prompt(2)
        lines = result.split("\n")
        assert len(lines) == 3  # header + 2 turns


# ── pop_last_n ────────────────────────────────────────────────────────────────


class TestPopLastN:
    def test_removes_last_n(self) -> None:
        h = _make()
        _fill(h, 5)
        h.pop_last_n(2)
        assert h.count() == 3
        assert [t.content for t in h.get_recent(3)] == ["msg 0", "msg 1", "msg 2"]

    def test_pop_all(self) -> None:
        h = _make()
        _fill(h, 3)
        h.pop_last_n(3)
        assert h.count() == 0

    def test_pop_more_than_count_clears(self) -> None:
        h = _make()
        _fill(h, 2)
        h.pop_last_n(10)
        assert h.count() == 0

    def test_pop_zero_is_noop(self) -> None:
        h = _make()
        _fill(h, 3)
        h.pop_last_n(0)
        assert h.count() == 3

    def test_pop_negative_is_noop(self) -> None:
        h = _make()
        _fill(h, 3)
        h.pop_last_n(-1)
        assert h.count() == 3


# ── max_turns cap ────────────────────────────────────────────────────────────


class TestMaxTurnsCap:
    def test_turns_capped_at_max(self) -> None:
        h = _make(max_turns=3)
        _fill(h, 5)
        assert h.count() == 3

    def test_oldest_turns_dropped(self) -> None:
        h = _make(max_turns=3)
        _fill(h, 5)
        contents = [t.content for t in h.get_recent(3)]
        assert contents == ["msg 2", "msg 3", "msg 4"]

    def test_exact_max_turns(self) -> None:
        h = _make(max_turns=3)
        _fill(h, 3)
        assert h.count() == 3
        assert h.get_recent(3)[0].content == "msg 0"

    def test_one_over_max(self) -> None:
        h = _make(max_turns=3)
        _fill(h, 4)
        assert h.count() == 3
        assert h.get_recent(3)[0].content == "msg 1"


# ── save / load round-trip ───────────────────────────────────────────────────


class TestSaveLoad:
    def test_round_trip(self, tmp_dir: Path) -> None:
        path = tmp_dir / "history.json"
        h = _make(path)
        h.add_turn("user", "hello")
        h.add_turn("character", "world")
        h.save()

        h2 = _make(path)
        h2.load()
        assert h2.count() == 2
        assert h2.get_recent(2)[0].content == "hello"
        assert h2.get_recent(2)[1].content == "world"

    def test_round_trip_preserves_roles(self, tmp_dir: Path) -> None:
        path = tmp_dir / "history.json"
        h = _make(path)
        h.add_turn("user", "q")
        h.add_turn("character", "a")
        h.save()

        h2 = _make(path)
        h2.load()
        turns = h2.get_recent(2)
        assert turns[0].role == "user"
        assert turns[1].role == "character"

    def test_save_creates_parent_dirs(self, tmp_dir: Path) -> None:
        path = tmp_dir / "deep" / "nested" / "history.json"
        h = _make(path)
        h.add_turn("user", "test")
        h.save()
        assert path.exists()

    def test_load_missing_file_is_noop(self, tmp_dir: Path) -> None:
        path = tmp_dir / "nonexistent.json"
        h = _make(path)
        h.load()
        assert h.count() == 0

    def test_save_without_path_is_noop(self) -> None:
        h = _make()
        h.add_turn("user", "x")
        h.save()  # should not raise
        assert h.count() == 1

    def test_load_without_path_is_noop(self) -> None:
        h = _make()
        h.load()  # should not raise

    def test_json_format_utf8(self, tmp_dir: Path) -> None:
        path = tmp_dir / "history.json"
        h = _make(path)
        h.add_turn("user", "한글 내용")
        h.save()

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["turns"][0]["content"] == "한글 내용"
