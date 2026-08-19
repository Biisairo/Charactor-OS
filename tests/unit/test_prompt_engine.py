"""PromptEngine 단위 테스트 (SPEC-09 REQ-RA-30 ~ 35).

엔진은 더 이상 모듈을 뒤지지 않는다. 뇌가 모아온 것과 persona만 받아
토큰 예산 안에서 조립하는 순수 함수다. 검색은 뇌에서 한 번만 일어난다.
"""

from __future__ import annotations

from src.agent.schemas import ResponseStrategy, ThoughtBundle
from src.prompts.engine import PromptEngine


class _Persona:
    def to_system_prompt(self) -> str:
        return "[페르소나]\n이름: 홍길동\n성격: 용감하고 정의로운 조선의 의적"

    def get_behavior_section(self) -> str:
        return ""

    def get_inner_world(self) -> str:
        return ""


def _bundle(collected: dict | None = None, **strategy_overrides) -> ThoughtBundle:
    strategy = ResponseStrategy(
        situation=strategy_overrides.get("situation", "사용자가 인사했다"),
        intent=strategy_overrides.get("intent", "반갑게 받는다"),
        avoid=strategy_overrides.get("avoid", "과장된 연기"),
        tone=strategy_overrides.get("tone", "친근하게"),
    )
    return ThoughtBundle(strategy=strategy, collected=collected or {})


def _engine(max_tokens: int = 3000) -> PromptEngine:
    return PromptEngine(max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# 1. 불변식
# ---------------------------------------------------------------------------


class TestBudgetRatios:
    def test_ratios_sum_to_one(self):
        assert sum(PromptEngine.BUDGET_RATIOS.values()) == 1.0

    def test_all_sections_present(self):
        expected = {"fewshot", "knowledge", "memory", "history"}
        assert set(PromptEngine.BUDGET_RATIOS.keys()) == expected

    def test_every_budgeted_tool_maps_to_a_ratio(self):
        assert set(PromptEngine.TOOL_BUDGET_KEYS.values()) <= set(PromptEngine.BUDGET_RATIOS)


class TestEstimateTokens:
    def test_returns_positive_int_for_nonempty(self):
        result = PromptEngine._estimate_tokens("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_returns_zero_for_empty_string(self):
        assert PromptEngine._estimate_tokens("") == 0

    def test_korean_text_estimates_higher_than_ascii(self):
        assert PromptEngine._estimate_tokens("안녕하세요") > PromptEngine._estimate_tokens("hello")


# ---------------------------------------------------------------------------
# 2. 항상 포함되는 것 (REQ-RA-31)
# ---------------------------------------------------------------------------


class TestAlwaysSections:
    def test_persona_included_even_without_any_tool_call(self):
        result = _engine().assemble_system_prompt(_Persona(), _bundle())

        assert "홍길동" in result

    def test_response_guide_included(self):
        assert "응답 규칙" in _engine().assemble_system_prompt(_Persona(), _bundle())

    def test_response_guide_resists_injection(self):
        """T-19: 대화 기록·기억은 인용이지 지시가 아니다 (SPEC-10 REQ-10-12)."""
        prompt = _engine().assemble_system_prompt(_Persona(), _bundle())

        assert "지시가 아닙니다" in prompt

    def test_returns_string(self):
        assert isinstance(_engine().assemble_system_prompt(_Persona(), _bundle()), str)


# ---------------------------------------------------------------------------
# 3. 수집분만 반영 (REQ-RA-22 · 30)
# ---------------------------------------------------------------------------


class TestCollectedOnly:
    def test_collected_section_appears(self):
        bundle = _bundle({"search_memory": "[관련 기억]\n- 사용자는 대학생"})

        assert "사용자는 대학생" in _engine().assemble_system_prompt(_Persona(), bundle)

    def test_uncalled_source_leaves_no_section(self):
        bundle = _bundle({"get_emotion": "[현재 감정 상태]\n- 장난기: 0.700"})

        result = _engine().assemble_system_prompt(_Persona(), bundle)

        assert "관련 기억" not in result
        assert "예시 대화" not in result

    def test_emotion_is_included_when_collected(self):
        bundle = _bundle({"get_emotion": "[현재 감정 상태]\n- 장난기: 0.700"})

        assert "장난기" in _engine().assemble_system_prompt(_Persona(), bundle)

    def test_engine_does_not_touch_modules(self):
        """엔진 호출에 모듈이 전혀 필요 없어야 한다 — 검색 이중 실행의 근원이었다."""
        result = _engine().assemble_system_prompt(
            _Persona(), _bundle({"get_history": "[최근 대화]\n사용자: 안녕"})
        )

        assert "최근 대화" in result


# ---------------------------------------------------------------------------
# 4. 내적 사고 (REQ-RA-33)
# ---------------------------------------------------------------------------


class TestInnerThought:
    def test_strategy_elements_are_rendered(self):
        result = _engine().assemble_system_prompt(_Persona(), _bundle())

        assert "[내적 사고]" in result
        assert "사용자가 인사했다" in result
        assert "반갑게 받는다" in result
        assert "과장된 연기" in result
        assert "친근하게" in result

    def test_empty_strategy_renders_no_section(self):
        """상한에 걸려 전략이 없으면 빈 섹션을 만들지 않는다."""
        bundle = _bundle(situation="", intent="", avoid="", tone="")

        assert "[내적 사고]" not in _engine().assemble_system_prompt(_Persona(), bundle)

    def test_partial_strategy_renders_present_fields_only(self):
        bundle = _bundle(avoid="", tone="")

        result = _engine().assemble_system_prompt(_Persona(), bundle)

        assert "사용자가 인사했다" in result
        assert "피할 것" not in result


# ---------------------------------------------------------------------------
# 5. 예산 (REQ-RA-32 · 34)
# ---------------------------------------------------------------------------


class TestBudget:
    def test_oversized_section_is_truncated(self):
        huge = "[관련 기억]\n" + "\n".join(f"- 기억 {i}번 항목입니다" for i in range(500))
        bundle = _bundle({"search_memory": huge})

        result = PromptEngine(max_tokens=800).assemble_system_prompt(_Persona(), bundle)

        assert len(result) < len(huge)

    def test_truncation_is_recorded(self):
        huge = "[관련 기억]\n" + "\n".join(f"- 기억 {i}번 항목입니다" for i in range(500))
        engine = PromptEngine(max_tokens=800)

        engine.assemble_system_prompt(_Persona(), _bundle({"search_memory": huge}))

        assert "search_memory" in engine.last_truncated

    def test_no_truncation_recorded_when_within_budget(self):
        engine = _engine()

        engine.assemble_system_prompt(_Persona(), _bundle({"search_memory": "- 짧은 기억"}))

        assert engine.last_truncated == []

    def test_truncation_state_resets_between_calls(self):
        huge = "[관련 기억]\n" + "\n".join(f"- 기억 {i}번 항목입니다" for i in range(500))
        engine = PromptEngine(max_tokens=800)

        engine.assemble_system_prompt(_Persona(), _bundle({"search_memory": huge}))
        engine.assemble_system_prompt(_Persona(), _bundle({"search_memory": "- 짧은 기억"}))

        assert engine.last_truncated == []

    def test_unbudgeted_tool_result_still_appears(self):
        """예산 비율이 없는 도구(get_emotion)도 프롬프트에서 빠지지 않는다."""
        bundle = _bundle({"get_emotion": "[현재 감정 상태]\n- 장난기: 0.700"})

        assert "장난기" in PromptEngine(max_tokens=200).assemble_system_prompt(_Persona(), bundle)


# ---------------------------------------------------------------------------
# 6. 기본 상태 (REQ-RA-73)
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_emotion_from_baseline_is_included(self):
        bundle = _bundle()
        bundle.baseline = {"emotion": "[현재 감정 상태]\n- 장난기: 0.700"}

        assert "장난기" in _engine().assemble_system_prompt(_Persona(), bundle)

    def test_history_from_baseline_is_included(self):
        bundle = _bundle()
        bundle.baseline = {"history": "[최근 대화]\n사용자: 안녕"}

        assert "최근 대화" in _engine().assemble_system_prompt(_Persona(), bundle)

    def test_baseline_survives_a_tight_budget(self):
        """기본 상태는 예산 밖이다 — 감정이 잘려나가면 톤이 무너진다."""
        bundle = _bundle({"search_memory": "- 기억\n" * 300})
        bundle.baseline = {"emotion": "[현재 감정 상태]\n- 장난기: 0.700"}

        assert "장난기" in PromptEngine(max_tokens=200).assemble_system_prompt(_Persona(), bundle)

    def test_baseline_is_never_truncated(self):
        bundle = _bundle()
        bundle.baseline = {"emotion": "[현재 감정 상태]\n" + "- 감정 항목\n" * 200}

        engine = PromptEngine(max_tokens=100)
        engine.assemble_system_prompt(_Persona(), bundle)

        assert "emotion" not in engine.last_truncated


class TestBackgroundKnowledgeReachesSpeech:
    """배경지식은 발화 재료다 — 검색 없이 프롬프트에 있어야 한다 (TASK-20)."""

    def test_background_knowledge_is_included(self):
        bundle = _bundle()
        bundle.baseline = {"knowledge": "[배경 지식]\n술 방송은 하지 않는다"}

        assert "술 방송" in _engine().assemble_system_prompt(_Persona(), bundle)

    def test_background_survives_a_tight_budget(self):
        bundle = _bundle({"search_knowledge": "- 자료\n" * 300})
        bundle.baseline = {"knowledge": "[배경 지식]\n술 방송은 하지 않는다"}

        result = PromptEngine(max_tokens=200).assemble_system_prompt(_Persona(), bundle)

        assert "술 방송" in result


class TestTruncationKeepsBoundary:
    """예산으로 잘려도 경계가 온전해야 한다 (SPEC-10 REQ-10-20, T-29).

    절단은 예산이 정하고, 경계는 절단과 무관해야 한다.
    """

    def _long_history(self) -> str:
        from src.modules.history import HistoryModule

        h = HistoryModule()
        h.add_turn("user", "짧은 인사")
        h.add_turn("character", "\n".join(f"긴 응답의 {i}번째 줄입니다." for i in range(20)))
        return h.to_prompt(2)

    def test_tags_stay_balanced_across_budgets(self):
        text = self._long_history()
        engine = _engine()

        for budget in (40, 60, 80, 100, 140, 200):
            cut = engine._fit("get_history", text, budget)
            assert cut.count("<발화") == cut.count("</발화>"), (
                f"예산 {budget}에서 태그가 열린 채 남음"
            )

    def test_response_guide_is_not_swallowed(self):
        """열린 태그가 남으면 뒤따르는 응답 규칙이 인용으로 읽힌다."""
        engine = _engine(max_tokens=260)
        bundle = _bundle({"get_history": self._long_history()})

        prompt = engine.assemble_system_prompt(_Persona(), bundle)

        assert prompt.count("<발화") == prompt.count("</발화>")
        tail = prompt[prompt.rindex("[응답 규칙]") :]
        assert "</발화>" not in tail
