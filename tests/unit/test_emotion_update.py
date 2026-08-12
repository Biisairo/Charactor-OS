"""감정 갱신의 현재 동작 고정 (TASK-14, REQ-14-2).

`EmotionModule.update()`는 143줄인데 착수 시점에 이를 직접 검증하는 테스트가
**0개**였다. 이 파일은 **특성화 테스트**다 — 지금 동작이 옳다고 주장하지 않고,
리팩터링이 동작을 바꾸지 않았음을 증명한다.

고정하는 동작 (순서가 중요하다):

1. `apply_decay()` — LLM 판단과 **무관하게** 먼저 적용된다
2. `_apply_triggers()` — 페르소나 키워드 트리거, 역시 LLM 이전
3. LLM 호출
4. `significant=false` → 이후 처리를 건너뛴다. **단 1·2는 이미 적용된 상태다**
5. `remove` 목록의 감정을 제거
6. 새 감정: 기존에 있으면 `old*0.7 + new*0.3`(소수 3자리 반올림), 없으면 그대로

LLM API 키 없이 결정론적으로 통과한다.
"""

from __future__ import annotations

from src.analysis import EmotionAnalysis
from src.modules.emotion import EmotionModule


class StubAnalyzer:
    """정해진 분석 결과를 돌려주는 스텁. **LLM 더블이 아니다** — REQ-14-4."""

    def __init__(self, analysis: EmotionAnalysis):
        self._analysis = analysis

    def analyze(self, user_input, character_response, current_emotions, history_context=""):
        return self._analysis


def _module(**kwargs) -> EmotionModule:
    return EmotionModule(save_path=None, **kwargs)


# ---------------------------------------------------------------------------
# LLM 이전 단계 — decay와 트리거는 판정과 무관하게 적용된다
# ---------------------------------------------------------------------------


class TestPreLlmStages:
    def test_decay_applies_even_when_not_significant(self):
        module = _module(decay_rate=0.1)
        module._emotions = {"슬픔": 0.5}

        module.update("안녕", "반갑네", StubAnalyzer(EmotionAnalysis(significant=False)))

        assert module.get_state()["슬픔"] == 0.45, "decay는 LLM 판정보다 먼저다"

    def test_decay_drops_negligible_emotions(self):
        module = _module(decay_rate=0.5)
        module._emotions = {"슬픔": 0.1}

        module.update("안녕", "반갑네", StubAnalyzer(EmotionAnalysis(significant=False)))

        assert "슬픔" not in module.get_state()

    def test_trigger_fires_before_llm(self):
        module = _module()
        module.set_triggers([{"keyword": "아버지", "emotion": "분노", "intensity": 0.6}])

        module.update(
            "아버지 얘기를 하고 싶어", "...", StubAnalyzer(EmotionAnalysis(significant=False))
        )

        assert module.get_state()["분노"] == 0.6


# ---------------------------------------------------------------------------
# significant=false — 이후 처리를 건너뛴다
# ---------------------------------------------------------------------------


class TestNotSignificant:
    def test_ignores_emotions_payload(self):
        module = _module()
        module.update(
            "안녕",
            "반갑네",
            StubAnalyzer(EmotionAnalysis(significant=False, emotions={"기쁨": 0.9})),
        )

        assert "기쁨" not in module.get_state()

    def test_ignores_remove_payload(self):
        module = _module(decay_rate=0.0)
        module._emotions = {"슬픔": 0.5}

        module.update(
            "안녕", "반갑네", StubAnalyzer(EmotionAnalysis(significant=False, remove=["슬픔"]))
        )

        assert module.get_state()["슬픔"] == 0.5


# ---------------------------------------------------------------------------
# significant=true — 제거와 블렌딩
# ---------------------------------------------------------------------------


class TestSignificant:
    def test_new_emotion_is_taken_as_is(self):
        module = _module(decay_rate=0.0)
        module.update(
            "슬픈 얘기",
            "저런",
            StubAnalyzer(EmotionAnalysis(significant=True, emotions={"연민": 0.8})),
        )

        assert module.get_state()["연민"] == 0.8

    def test_existing_emotion_is_blended(self):
        """기존 0.5, 새 값 1.0 → 0.5*0.7 + 1.0*0.3 = 0.65"""
        module = _module(decay_rate=0.0)
        module._emotions = {"연민": 0.5}

        module.update(
            "슬픈 얘기",
            "저런",
            StubAnalyzer(EmotionAnalysis(significant=True, emotions={"연민": 1.0})),
        )

        assert module.get_state()["연민"] == 0.65

    def test_blend_is_rounded_to_three_places(self):
        module = _module(decay_rate=0.0)
        module._emotions = {"연민": 0.3333}

        module.update(
            "x", "y", StubAnalyzer(EmotionAnalysis(significant=True, emotions={"연민": 0.1111}))
        )

        assert module.get_state()["연민"] == round(0.3333 * 0.7 + 0.1111 * 0.3, 3)

    def test_remove_deletes_emotion(self):
        module = _module(decay_rate=0.0)
        module._emotions = {"분노": 0.7}

        module.update(
            "화해했어", "다행이군", StubAnalyzer(EmotionAnalysis(significant=True, remove=["분노"]))
        )

        assert "분노" not in module.get_state()

    def test_remove_of_absent_emotion_is_harmless(self):
        module = _module(decay_rate=0.0)
        module._emotions = {"기쁨": 0.4}

        module.update(
            "x", "y", StubAnalyzer(EmotionAnalysis(significant=True, remove=["없는감정"]))
        )

        assert module.get_state() == {"기쁨": 0.4}

    def test_out_of_range_value_is_rejected(self):
        module = _module(decay_rate=0.0)
        module.update(
            "x", "y", StubAnalyzer(EmotionAnalysis(significant=True, emotions={"분노": 1.5}))
        )

        assert "분노" not in module.get_state()

    def test_non_numeric_value_is_rejected(self):
        module = _module(decay_rate=0.0)
        module.update(
            "x", "y", StubAnalyzer(EmotionAnalysis(significant=True, emotions={"분노": "높음"}))
        )

        assert "분노" not in module.get_state()


# ---------------------------------------------------------------------------
# 분석 결과가 비어 있는 경우
#
# 거부·파싱 실패를 "변화 없음"으로 바꾸는 일은 이제 분석 층의 책임이며
# `test_analysis.py`가 검증한다. 도메인은 significant=False만 안다.
# ---------------------------------------------------------------------------


class TestNoChange:
    def test_empty_analysis_keeps_state(self):
        module = _module(decay_rate=0.0)
        module._emotions = {"기쁨": 0.4}

        module.update("x", "y", StubAnalyzer(EmotionAnalysis()))

        assert module.get_state() == {"기쁨": 0.4}
