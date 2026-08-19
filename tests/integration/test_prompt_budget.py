"""예산 내역이 debug 플래그와 무관하게 드러난다 (SPEC-11 T-13 ~ T-15).

종전에는 절단 사실이 `_log()`를 통해 메모리 버퍼에만 쌓이고 `debug=True`일
때만 출력됐다. 평가 하네스는 `debug=False`로 돌므로, 소민찌가 검색 결과 4종을
통째로 잃고도 모든 평가가 그 사실을 보지 못한 채 측정됐다 (P-4).

이 테스트는 LLM API 키 없이 결정론적으로 통과한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.call_log import CallRecord
from src.prompts.engine import PromptEngine
from tests.conftest import PipelineMockClient, make_character_os

VALID = "흠, 그리 말하니 반갑구려."


class _CapturingLogger:
    """턴 기록을 메모리에 모으는 로거. 파일에 쓰지 않는다."""

    capture_payload = False

    def __init__(self) -> None:
        self.turns: list[dict] = []

    def log_call(self, record: CallRecord, **kwargs) -> None:
        pass

    def log_turn(self, **kwargs) -> None:
        self.turns.append(kwargs)


class _Counter:
    """글자 수를 그대로 토큰 수로 센다."""

    method = "test-counter"
    fallback_reason = ""

    def count(self, text: str) -> int:
        return len(text or "")


@pytest.fixture
def logged_cos(character_dir: Path, tmp_path: Path):
    logger = _CapturingLogger()
    cos = make_character_os(
        character_dir,
        tmp_path / "state",
        PipelineMockClient(response=VALID),
        call_logger=logger,
        prompt_engine=PromptEngine(max_tokens=100_000, counter=_Counter()),
    )
    return cos, logger


class TestBudgetReachesOpsLog:
    def test_turn_record_carries_budget(self, logged_cos):
        """T-13 · REQ-11-9: debug=False에서도 남는다."""
        cos, logger = logged_cos
        assert cos._debug is False

        cos.chat("안녕하시오")

        budget = logger.turns[-1]["extra"]["prompt_budget"]
        assert budget["max_tokens"] == 100_000
        assert budget["fixed_tokens"] > 0
        assert budget["method"] == "test-counter"
        assert budget["starved"] is False


class TestStarvedIsVisible:
    def test_user_sees_warning(self, character_dir: Path, tmp_path: Path):
        """T-14 · REQ-11-8: 검색 몫이 없으면 사용자에게 보인다."""
        seen: list[str] = []
        cos = make_character_os(
            character_dir,
            tmp_path / "state",
            PipelineMockClient(response=VALID),
            output=seen.append,
            call_logger=_CapturingLogger(),
            # 고정 섹션만으로 소진되는 예산
            prompt_engine=PromptEngine(max_tokens=200, counter=_Counter()),
        )

        cos.chat("안녕하시오")

        assert any("예산이 부족" in msg for msg in seen)

    def test_starved_is_recorded(self, character_dir: Path, tmp_path: Path):
        logger = _CapturingLogger()
        cos = make_character_os(
            character_dir,
            tmp_path / "state",
            PipelineMockClient(response=VALID),
            output=lambda _m: None,
            call_logger=logger,
            prompt_engine=PromptEngine(max_tokens=200, counter=_Counter()),
        )

        cos.chat("안녕하시오")

        assert logger.turns[-1]["extra"]["prompt_budget"]["starved"] is True


class TestBudgetReachesTrace:
    def test_trace_details_carry_budget(self, character_dir: Path, tmp_path: Path):
        """T-15 · REQ-11-10."""
        cos = make_character_os(
            character_dir,
            tmp_path / "state",
            PipelineMockClient(response=VALID),
            call_logger=_CapturingLogger(),
            trace=True,
            prompt_engine=PromptEngine(max_tokens=100_000, counter=_Counter()),
        )

        cos.chat("안녕하시오")

        stage = next(s for s in cos._last_trace.stages if s.name == "response")
        assert "예산" in stage.details["prompt_budget"]


class TestSectionLimitsAgainstRealAssets:
    """상한은 실측에 근거한다 (SPEC-11 결정 6, T-19).

    현행 자산이 상한에 걸리면 경보가 늘 울려 무의미해진다. 근거가 실측임을
    여기서 고정한다.
    """

    @pytest.mark.parametrize("character", ["hong-gil-dong", "han-so-min"])
    def test_shipped_assets_pass_every_limit(self, character: str, tmp_path: Path):
        from src.character_layout import CharacterLayout
        from src.modules.knowledge import KnowledgeModule
        from src.modules.persona import PersonaModule

        layout = CharacterLayout.of(Path("characters") / character)
        persona = PersonaModule(str(layout.persona_path))
        persona.load()
        knowledge = KnowledgeModule(str(layout.knowledge_dir), embedding_fn=lambda _t: None)
        knowledge.load_all()

        bundle = _real_bundle(knowledge.base_text())
        engine = PromptEngine(max_tokens=3000)

        engine.assemble_system_prompt(persona, bundle)

        assert engine.last_report.over_limit == []


def _real_bundle(base_text: str):
    from src.agent.schemas import ResponseStrategy, ThoughtBundle

    bundle = ThoughtBundle(strategy=ResponseStrategy())
    bundle.baseline = {"knowledge": base_text, "emotion": "", "history": ""}
    return bundle


class TestOverLimitIsVisible:
    """T-20 · REQ-11-16: 합계만 알리면 무엇을 줄일지 모른다."""

    def test_user_sees_offending_section_by_name(self, character_dir: Path, tmp_path: Path):
        seen: list[str] = []
        cos = make_character_os(
            character_dir,
            tmp_path / "state",
            PipelineMockClient(response=VALID),
            output=seen.append,
            call_logger=_CapturingLogger(),
            # 페르소나가 상한(40%)을 넘도록 예산을 조인다
            prompt_engine=PromptEngine(max_tokens=300, counter=_Counter()),
        )

        cos.chat("안녕하시오")

        warning = next((m for m in seen if "상한을 넘은" in m), "")
        assert "persona" in warning

    def test_sections_reach_the_ops_log(self, character_dir: Path, tmp_path: Path):
        logger = _CapturingLogger()
        cos = make_character_os(
            character_dir,
            tmp_path / "state",
            PipelineMockClient(response=VALID),
            output=lambda _m: None,
            call_logger=logger,
            prompt_engine=PromptEngine(max_tokens=300, counter=_Counter()),
        )

        cos.chat("안녕하시오")
        budget = logger.turns[-1]["extra"]["prompt_budget"]

        assert budget["sections"]["persona"] > 0
        assert "persona" in budget["over_limit"]
