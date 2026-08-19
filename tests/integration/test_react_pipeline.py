"""뇌가 파이프라인에 실제로 연결되었는지 검증한다 (SPEC-09).

단위 테스트가 뇌 내부를 본다면, 여기서는 뇌가 Stage 2·3과 맞물려
프롬프트를 지배하고, 실패를 숨기지 않고, 작업기억을 남기는지를 본다.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.call_log import CallLogger
from src.character_layout import CharacterLayout
from tests.conftest import (
    DEFAULT_BRAIN_STRATEGY,
    PipelineMockClient,
    finish_call,
    make_character_os,
    text_step,
    tool_call,
    tool_step,
)


def _cos(character_dir: Path, tmp_path: Path, client: PipelineMockClient, **overrides):
    return make_character_os(character_dir, tmp_path, client, **overrides)


# ---------------------------------------------------------------------------
# 1. 뇌가 프롬프트를 지배한다 (REQ-RA-30 ~ 35)
# ---------------------------------------------------------------------------


class TestBrainDrivesPrompt:
    def test_strategy_reaches_response_prompt(self, character_dir, tmp_path):
        client = PipelineMockClient()
        cos = _cos(character_dir, tmp_path, client)

        cos.chat("안녕")

        response_prompts = [
            r["messages"][0]["content"]
            for r in client.all_call_records
            if r["messages"] and r["messages"][0].get("role") == "system"
        ]
        assert any(DEFAULT_BRAIN_STRATEGY["intent"] in p for p in response_prompts)

    def test_uncalled_source_is_absent_from_prompt(self, character_dir, tmp_path):
        """뇌가 few-shot을 부르지 않았으면 프롬프트에도 예시가 없다."""
        client = PipelineMockClient(brain_script=[tool_step(finish_call(**DEFAULT_BRAIN_STRATEGY))])
        cos = _cos(character_dir, tmp_path, client)

        cos.chat("안녕")

        system_prompt = next(
            r["messages"][0]["content"]
            for r in client.all_call_records
            if r["messages"]
            and r["messages"][0].get("role") == "system"
            and "응답 규칙" in str(r["messages"][0]["content"])
        )
        assert "예시 대화" not in system_prompt

    def test_search_runs_once_per_turn(self, character_dir, tmp_path, monkeypatch):
        """검색의 단일 실행 지점은 뇌다 — 엔진이 다시 뒤지지 않는다 (REQ-RA-35)."""
        client = PipelineMockClient(
            brain_script=[
                tool_step(tool_call("search_knowledge", query="활빈당")),
                tool_step(finish_call(**DEFAULT_BRAIN_STRATEGY)),
            ]
        )
        cos = _cos(character_dir, tmp_path, client)

        calls: list[str] = []
        original = cos.knowledge.search_relevant
        monkeypatch.setattr(
            cos.knowledge,
            "search_relevant",
            lambda query, token_budget=500: (calls.append(query), original(query, token_budget))[1],
        )

        cos.chat("활빈당이 뭐야")

        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 2. 계측·트레이스 (REQ-RA-07 · 60 ~ 63)
# ---------------------------------------------------------------------------


class TestObservability:
    def test_brain_calls_are_labeled_react(self, character_dir, tmp_path):
        client = PipelineMockClient()
        cos = _cos(character_dir, tmp_path, client, trace=True)

        cos.chat("안녕")

        by_label = cos._last_trace.metrics["by_label"]
        assert by_label["react"]["calls"] == 2

    def test_trace_records_iterations_and_tools(self, character_dir, tmp_path):
        client = PipelineMockClient()
        cos = _cos(character_dir, tmp_path, client, trace=True)

        cos.chat("안녕")

        context_stage = next(s for s in cos._last_trace.stages if s.name == "context")
        assert context_stage.details["iterations"] == 2
        assert context_stage.details["hit_cap"] is False
        assert "search_memory" in context_stage.details["tools_used"]

    def test_trace_records_strategy(self, character_dir, tmp_path):
        client = PipelineMockClient()
        cos = _cos(character_dir, tmp_path, client, trace=True)

        cos.chat("안녕")

        context_stage = next(s for s in cos._last_trace.stages if s.name == "context")
        assert DEFAULT_BRAIN_STRATEGY["situation"] in str(context_stage.details["strategy"])

    def test_brain_call_count_never_exceeds_cap(self, character_dir, tmp_path):
        client = PipelineMockClient(
            brain_script=[tool_step(tool_call("get_history", n=i + 1)) for i in range(20)]
        )
        cos = _cos(character_dir, tmp_path, client, trace=True)

        cos.chat("안녕")

        assert cos._last_trace.metrics["by_label"]["react"]["calls"] <= 5


# ---------------------------------------------------------------------------
# 3. 상한 도달은 실패가 아니다 (REQ-RA-04 · 42)
# ---------------------------------------------------------------------------


class TestCapIsNotFailure:
    def test_response_is_still_produced(self, character_dir, tmp_path):
        client = PipelineMockClient(
            brain_script=[tool_step(tool_call("get_history", n=i + 1)) for i in range(20)]
        )
        cos = _cos(character_dir, tmp_path, client)

        assert cos.chat("복잡한 질문") is not None

    def test_hit_cap_is_visible_in_trace(self, character_dir, tmp_path):
        client = PipelineMockClient(
            brain_script=[tool_step(tool_call("get_history", n=i + 1)) for i in range(20)]
        )
        cos = _cos(character_dir, tmp_path, client, trace=True)

        cos.chat("복잡한 질문")

        context_stage = next(s for s in cos._last_trace.stages if s.name == "context")
        assert context_stage.details["hit_cap"] is True

    def test_turn_is_persisted(self, character_dir, tmp_path):
        client = PipelineMockClient(
            brain_script=[tool_step(tool_call("get_history", n=i + 1)) for i in range(20)]
        )
        cos = _cos(character_dir, tmp_path, client)

        cos.chat("복잡한 질문")

        assert cos.history.count() == 2


# ---------------------------------------------------------------------------
# 4. 실패는 숨기지 않는다 (REQ-RA-40 ~ 44)
# ---------------------------------------------------------------------------


class TestFailureSurfaces:
    def test_brain_llm_error_kills_the_turn(self, character_dir, tmp_path):
        client = PipelineMockClient(brain_script=[RuntimeError("연결 끊김")])
        cos = _cos(character_dir, tmp_path, client)

        assert cos.chat("안녕") is None

    def test_failed_turn_leaves_no_state(self, character_dir, tmp_path):
        client = PipelineMockClient(brain_script=[RuntimeError("연결 끊김")])
        cos = _cos(character_dir, tmp_path, client)

        cos.chat("안녕")

        assert cos.history.count() == 0
        assert cos.memory.snapshot_count() == 0

    def test_brain_refusal_kills_the_turn(self, character_dir, tmp_path):
        client = PipelineMockClient(
            brain_script=[text_step("Your request was rejected by the provider")]
        )
        cos = _cos(character_dir, tmp_path, client)

        assert cos.chat("안녕") is None

    def test_bad_finish_kills_the_turn(self, character_dir, tmp_path):
        client = PipelineMockClient(
            brain_script=[tool_step(tool_call("finish", raw_arguments="{깨짐"))]
        )
        cos = _cos(character_dir, tmp_path, client)

        assert cos.chat("안녕") is None

    def test_failure_reason_is_classified_in_trace(self, character_dir, tmp_path):
        client = PipelineMockClient(brain_script=[RuntimeError("연결 끊김")])
        cos = _cos(character_dir, tmp_path, client, trace=True)

        cos.chat("안녕")

        assert "llm_error" in cos._last_trace.error


# ---------------------------------------------------------------------------
# 5. 작업기억 (REQ-RA-50 ~ 57)
# ---------------------------------------------------------------------------


def _script_with_thoughts(**finish_kwargs):
    return [tool_step(finish_call(**DEFAULT_BRAIN_STRATEGY, **finish_kwargs))]


class TestWorkingMemoryIntegration:
    def test_new_thought_is_persisted(self, character_dir, tmp_path):
        client = PipelineMockClient(
            brain_script=_script_with_thoughts(
                new_thoughts=[{"kind": "question", "content": "왜 말수가 줄었을까"}]
            )
        )
        cos = _cos(character_dir, tmp_path, client)

        cos.chat("...")

        saved = json.loads((tmp_path / "working_memory.json").read_text(encoding="utf-8"))
        assert saved["items"][0]["content"] == "왜 말수가 줄었을까"

    def test_thought_reaches_next_turn_prompt(self, character_dir, tmp_path):
        client = PipelineMockClient(
            brain_script=_script_with_thoughts(
                new_thoughts=[{"kind": "question", "content": "왜 말수가 줄었을까"}]
            )
        )
        cos = _cos(character_dir, tmp_path, client)
        cos.chat("...")

        client._brain_script_source = [tool_step(finish_call(**DEFAULT_BRAIN_STRATEGY))]
        client.all_call_records.clear()
        cos.chat("응")

        brain_prompts = [
            str(r["messages"][0]["content"])
            for r in client.all_call_records
            if r["messages"] and r["messages"][0].get("role") == "system"
        ]
        assert any("왜 말수가 줄었을까" in p for p in brain_prompts)

    def test_failed_turn_rolls_back_working_memory(self, character_dir, tmp_path):
        client = PipelineMockClient(
            brain_script=_script_with_thoughts(
                new_thoughts=[{"kind": "question", "content": "왜 말수가 줄었을까"}]
            ),
            fail_when=lambda text: "감정 상태를 업데이트하세요" in text,
        )
        cos = _cos(character_dir, tmp_path, client)

        cos.chat("...")

        assert cos.working_memory.count() == 0
        assert not (tmp_path / "working_memory.json").exists()

    def test_state_lives_under_character_state_dir(self, character_dir):
        layout = CharacterLayout.of(character_dir)

        assert layout.working_memory_path.parent == layout.state_dir


# ---------------------------------------------------------------------------
# 6. 검토 기준과 추적 (TASK-19)
# ---------------------------------------------------------------------------


class TestReviewCriteriaComeFromCharacter:
    def test_review_prompt_states_the_character_era(self, character_dir, tmp_path):
        client = PipelineMockClient()
        cos = _cos(character_dir, tmp_path, client, no_review=False)

        prompt = cos.reviewer._build_review_prompt("안녕", "초안")

        assert "조선" in prompt

    def test_review_prompt_has_no_hardcoded_era_example(self, character_dir, tmp_path):
        """다른 캐릭터의 시대를 예시로 박아두면 정상 응답이 FAIL된다."""
        client = PipelineMockClient()
        cos = _cos(character_dir, tmp_path, client, no_review=False)

        assert "한양" not in cos.reviewer._build_review_prompt("안녕", "초안")


class TestReviewStatsAreLogged:
    def test_turn_log_carries_verdicts(self, character_dir, tmp_path):
        recorded: list[dict] = []

        class _CapturingLogger(CallLogger):
            def log_turn(self, **kwargs):
                recorded.append(kwargs)

        client = PipelineMockClient(response='{"verdict": "PASS"}')
        cos = _cos(
            character_dir,
            tmp_path,
            client,
            no_review=False,
            call_logger=_CapturingLogger(enabled=False),
        )

        cos.chat("안녕")

        assert "review_verdicts" in recorded[-1]["extra"]
        assert "regenerations" in recorded[-1]["extra"]


# ---------------------------------------------------------------------------
# 7. 배경지식과 일반지식 (TASK-20)
# ---------------------------------------------------------------------------


class TestKnowledgeSplit:
    def _seed(self, character_dir: Path) -> None:
        knowledge = CharacterLayout.of(character_dir).knowledge_dir
        (knowledge / "base").mkdir(parents=True, exist_ok=True)
        (knowledge / "base" / "creed.md").write_text(
            "# 신조\n\n가난한 자의 것은 빼앗지 않는다.", encoding="utf-8"
        )
        (knowledge / "general").mkdir(parents=True, exist_ok=True)
        (knowledge / "general" / "hideout.md").write_text(
            "# 은신처\n\n## 산채 위치\n\n험한 골짜기 안쪽에 있다.", encoding="utf-8"
        )

    def test_background_reaches_the_brain(self, character_dir, tmp_path):
        self._seed(character_dir)
        client = PipelineMockClient()
        cos = _cos(character_dir, tmp_path, client)

        cos.chat("안녕")

        brain_prompt = client.all_call_records[0]["messages"][0]["content"]
        assert "가난한 자의 것은 빼앗지 않는다" in brain_prompt

    def test_background_reaches_the_response_prompt(self, character_dir, tmp_path):
        self._seed(character_dir)
        client = PipelineMockClient()
        cos = _cos(character_dir, tmp_path, client)

        cos.chat("안녕")

        response_prompt = next(
            r["messages"][0]["content"]
            for r in client.all_call_records
            if r["messages"] and "[응답 규칙]" in str(r["messages"][0].get("content", ""))
        )
        assert "가난한 자의 것은 빼앗지 않는다" in response_prompt

    def test_general_knowledge_is_only_an_index_until_searched(self, character_dir, tmp_path):
        self._seed(character_dir)
        client = PipelineMockClient()
        cos = _cos(character_dir, tmp_path, client)

        cos.chat("안녕")

        brain_prompt = client.all_call_records[0]["messages"][0]["content"]
        assert "산채 위치" in brain_prompt
        assert "험한 골짜기" not in brain_prompt

    def test_search_tool_reaches_general_knowledge(self, character_dir, tmp_path):
        self._seed(character_dir)
        client = PipelineMockClient(
            brain_script=[
                tool_step(tool_call("search_knowledge", query="산채")),
                tool_step(finish_call(**DEFAULT_BRAIN_STRATEGY)),
            ]
        )
        cos = _cos(character_dir, tmp_path, client)

        cos.chat("산채가 어디 있소?")

        response_prompt = next(
            r["messages"][0]["content"]
            for r in client.all_call_records
            if r["messages"] and "[응답 규칙]" in str(r["messages"][0].get("content", ""))
        )
        assert "험한 골짜기" in response_prompt
