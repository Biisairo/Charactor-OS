"""PromptEngine 단위 테스트."""

from __future__ import annotations

from src.prompts.engine import PromptEngine

# ---------------------------------------------------------------------------
# Minimal mock objects — only the methods PromptEngine actually calls
# ---------------------------------------------------------------------------


class _Persona:
    """Minimal persona stub."""

    def to_system_prompt(self) -> str:
        return "[페르소나]\n이름: 홍길동\n성격: 용감하고 정의로운 조선의 의적"

    def get_behavior_section(self) -> str:
        return ""

    def get_inner_world(self) -> str:
        return ""

    def get_relationships(self) -> list[dict]:
        return []


class _Emotion:
    """Minimal emotion stub."""

    def to_prompt(self) -> str:
        return "[감정 상태]\n기쁨: 0.6, 슬픔: 0.1"

    def get_state(self) -> dict:
        return {"joy": 0.6, "sadness": 0.1}


class _Memory:
    """Minimal memory stub."""

    def to_prompt(self, *, query: str, top_k: int, token_budget: int) -> str:
        return ""


class _Knowledge:
    """Minimal knowledge stub."""

    def search_relevant(self, *, query: str, token_budget: int) -> str:
        return ""

    def get_relationships(self) -> list[dict]:
        return []


class _History:
    """Minimal history stub."""

    def to_prompt(self, *, n: int) -> str:
        return ""


class _FewShot:
    """Minimal few-shot stub."""

    def to_prompt(self, *, query: str, emotions: dict, top_k: int, token_budget: int) -> str:
        return ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBudgetRatios:
    """BUDGET_RATIOS 불변식."""

    def test_ratios_sum_to_one(self):
        assert sum(PromptEngine.BUDGET_RATIOS.values()) == 1.0

    def test_all_sections_present(self):
        expected = {"fewshot", "knowledge", "relationships", "memory", "history"}
        assert set(PromptEngine.BUDGET_RATIOS.keys()) == expected


class TestEstimateTokens:
    """_estimate_tokens 동작."""

    def test_returns_positive_int_for_nonempty(self):
        result = PromptEngine._estimate_tokens("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_returns_zero_for_empty_string(self):
        assert PromptEngine._estimate_tokens("") == 0

    def test_korean_text_estimates_higher_than_ascii(self):
        ascii_text = "hello"
        korean_text = "안녕하세요"
        # Korean chars get 1.5 tokens each vs ASCII 0.3 — same char count
        assert PromptEngine._estimate_tokens(korean_text) > PromptEngine._estimate_tokens(
            ascii_text
        )


class TestAssembleSystemPrompt:
    """assemble_system_prompt 통합."""

    def _make_engine(self) -> PromptEngine:
        return PromptEngine(max_tokens=3000)

    def test_returns_string(self):
        result = self._make_engine().assemble_system_prompt(
            user_input="안녕하세요",
            persona=_Persona(),
            emotion=_Emotion(),
            memory=_Memory(),
            knowledge=_Knowledge(),
            history=_History(),
            fewshot=_FewShot(),
        )
        assert isinstance(result, str)

    def test_contains_persona_name(self):
        result = self._make_engine().assemble_system_prompt(
            user_input="안녕하세요",
            persona=_Persona(),
            emotion=_Emotion(),
            memory=_Memory(),
            knowledge=_Knowledge(),
            history=_History(),
            fewshot=_FewShot(),
        )
        assert "홍길동" in result

    def test_contains_emotion_section(self):
        result = self._make_engine().assemble_system_prompt(
            user_input="안녕하세요",
            persona=_Persona(),
            emotion=_Emotion(),
            memory=_Memory(),
            knowledge=_Knowledge(),
            history=_History(),
            fewshot=_FewShot(),
        )
        assert "감정 상태" in result

    def test_contains_response_guide(self):
        result = self._make_engine().assemble_system_prompt(
            user_input="안녕하세요",
            persona=_Persona(),
            emotion=_Emotion(),
            memory=_Memory(),
            knowledge=_Knowledge(),
            history=_History(),
            fewshot=_FewShot(),
        )
        assert "응답 규칙" in result

    def test_nonempty_sections_are_included(self):
        """모든 stub이 비어있어도 persona/emotion/response_guide는 포함된다."""
        result = self._make_engine().assemble_system_prompt(
            user_input="테스트",
            persona=_Persona(),
            emotion=_Emotion(),
            memory=_Memory(),
            knowledge=_Knowledge(),
            history=_History(),
            fewshot=_FewShot(),
        )
        # At minimum: persona, emotion, response guide
        assert "페르소나" in result
        assert "응답 규칙" in result
