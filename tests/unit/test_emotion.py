"""EmotionModule 단위 테스트."""

from __future__ import annotations

import pytest

from src.modules.emotion import EmotionModule

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make(save_path: str | None = None) -> EmotionModule:
    return EmotionModule(save_path=save_path)


# ---------------------------------------------------------------------------
# initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_empty_emotions(self):
        m = _make()
        assert m.get_state() == {}

    def test_no_triggers(self):
        m = _make()
        # _apply_triggers on empty triggers is a no-op
        m._apply_triggers("아버지")
        assert m.get_state() == {}


# ---------------------------------------------------------------------------
# set_triggers / _apply_triggers
# ---------------------------------------------------------------------------


class TestTriggers:
    TRIGGERS = [
        {"keyword": "아버지", "emotion": "분노", "intensity": 0.6},
        {"keyword": "어머니", "emotion": "그리움", "intensity": 0.7},
        {"keyword": "차별", "emotion": "분노", "intensity": 0.8},
    ]

    def test_set_triggers_stores_list(self):
        m = _make()
        m.set_triggers(self.TRIGGERS)
        assert len(m._triggers) == 3

    def test_set_triggers_empty_clears(self):
        m = _make()
        m.set_triggers(self.TRIGGERS)
        m.set_triggers([])
        assert m._triggers == []

    def test_apply_triggers_single_match(self):
        m = _make()
        m.set_triggers(self.TRIGGERS)
        m._apply_triggers("아버지가 보고 싶다")
        assert m.get_state()["분노"] == pytest.approx(0.6)

    def test_apply_triggers_multiple_keywords_same_emotion(self):
        """같은 감정(분노)에 대해 '아버지'와 '차별'이 둘 다 매칭되면 max 유지."""
        m = _make()
        m.set_triggers(self.TRIGGERS)
        m._apply_triggers("아버지와 차별에 대해")
        # '아버지' → 분노 0.6, then '차별' → 분노 max(0.6, 0.8) = 0.8
        assert m.get_state()["분노"] == pytest.approx(0.8)

    def test_apply_triggers_case_insensitive(self):
        m = _make()
        m.set_triggers([{"keyword": "hello", "emotion": "joy", "intensity": 0.5}])
        m._apply_triggers("HELLO world")
        assert m.get_state()["joy"] == pytest.approx(0.5)

    def test_apply_triggers_no_match(self):
        m = _make()
        m.set_triggers(self.TRIGGERS)
        m._apply_triggers("오늘 날씨가 좋다")
        assert m.get_state() == {}

    def test_apply_triggers_max_preserves_higher(self):
        """이미 높은 감정 값이 있으면 trigger의 낮은 값으로 덮어쓰지 않는다."""
        m = _make()
        m.set_triggers([{"keyword": "테스트", "emotion": "기쁨", "intensity": 0.3}])
        m._emotions["기쁨"] = 0.9
        m._apply_triggers("테스트입니다")
        assert m.get_state()["기쁨"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# apply_decay
# ---------------------------------------------------------------------------


class TestDecay:
    def test_decay_reduces_values(self):
        m = _make()
        m._emotions = {"행복": 1.0}
        m.apply_decay()
        # 1.0 * (1 - 0.1) = 0.9
        assert m.get_state()["행복"] == pytest.approx(0.9)

    def test_decay_removes_below_threshold(self):
        m = _make()
        # value * 0.9 <= 0.05 → removed
        m._emotions = {"미미한감정": 0.05}
        m.apply_decay()
        assert "미미한감정" not in m.get_state()

    def test_decay_keeps_above_threshold(self):
        m = _make()
        m._emotions = {"강한감정": 0.1}
        m.apply_decay()
        # 0.1 * 0.9 = 0.09 > 0.05
        assert m.get_state()["강한감정"] == pytest.approx(0.09)

    def test_decay_custom_rate(self):
        m = EmotionModule(decay_rate=0.5)
        m._emotions = {"감정": 1.0}
        m.apply_decay()
        assert m.get_state()["감정"] == pytest.approx(0.5)

    def test_decay_empty_state(self):
        m = _make()
        m.apply_decay()
        assert m.get_state() == {}


# ---------------------------------------------------------------------------
# snapshot / restore round-trip
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_round_trip(self):
        m = _make()
        m._emotions = {"분노": 0.7, "슬픔": 0.3}
        snap = m.snapshot()

        m._emotions["분노"] = 0.1
        m._emotions["기쁨"] = 0.9
        m.restore(snap)

        assert m.get_state() == {"분노": 0.7, "슬픔": 0.3}

    def test_snapshot_returns_copy(self):
        m = _make()
        m._emotions = {"행복": 0.5}
        snap = m.snapshot()
        snap["행복"] = 0.0
        assert m.get_state()["행복"] == pytest.approx(0.5)

    def test_restore_replaces_state(self):
        m = _make()
        m._emotions = {"기쁨": 0.8}
        m.restore({"슬픔": 0.6})
        assert m.get_state() == {"슬픔": 0.6}

    def test_snapshot_empty(self):
        m = _make()
        snap = m.snapshot()
        assert snap == {}


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self, tmp_dir):
        save_path = str(tmp_dir / "emotion.json")
        m = _make(save_path=save_path)
        m._emotions = {"분노": 0.7, "슬픔": 0.3}
        m.save()

        m2 = _make(save_path=save_path)
        m2.load()
        assert m2.get_state()["분노"] == pytest.approx(0.7)
        assert m2.get_state()["슬픔"] == pytest.approx(0.3)

    def test_save_creates_file(self, tmp_dir):
        save_path = str(tmp_dir / "emotion.json")
        m = _make(save_path=save_path)
        m._emotions = {"기쁨": 0.5}
        m.save()
        assert (tmp_dir / "emotion.json").exists()

    def test_load_missing_file_is_noop(self, tmp_dir):
        save_path = str(tmp_dir / "nonexistent.json")
        m = _make(save_path=save_path)
        m._emotions = {"행복": 0.5}
        m.load()
        assert m.get_state() == {"행복": 0.5}

    def test_save_without_path_is_noop(self):
        m = _make()
        m._emotions = {"기쁨": 0.5}
        # should not raise
        m.save()
        assert m.get_state() == {"기쁨": 0.5}

    def test_load_without_path_is_noop(self):
        m = _make()
        m.load()
        assert m.get_state() == {}

    def test_save_excludes_last_updated_from_emotions(self, tmp_dir):
        """save에 포함된 last_updated가 load 시 감정으로 복원되지 않는다."""
        save_path = str(tmp_dir / "emotion.json")
        m = _make(save_path=save_path)
        m._emotions = {"분노": 0.5}
        m.save()

        m2 = _make(save_path=save_path)
        m2.load()
        state = m2.get_state()
        assert "last_updated" not in state
        assert state["분노"] == pytest.approx(0.5)
