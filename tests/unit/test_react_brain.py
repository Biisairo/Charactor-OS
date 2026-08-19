"""ReAct 뇌 단위 테스트 (SPEC-09 REQ-RA-01 ~ 44).

뇌는 LLM 없이 검증 가능해야 한다 (NFR-RA-03). 클라이언트를 주입해
"몇 번째 호출에 무엇을 돌려줄지"를 대본으로 고정하고, 루프 제어·도구 실행·
실패 분류를 값으로 확인한다.
"""

from __future__ import annotations

import json

import pytest

from src.agent.brain import (
    BRAIN_MAX_OUTPUT_TOKENS,
    MAX_STRATEGY_CHARS,
    MAX_THOUGHT_CHARS,
    BrainError,
    ReActBrain,
)
from src.agent.tools import FINISH_TOOL, ToolRegistry
from src.llm.client import TrimmedMessage
from tests.conftest import ScriptedClient, finish_call, text_step, tool_call, tool_step

# ---------------------------------------------------------------------------
# 모듈 스텁 — 뇌가 도구로 부르는 표면만 갖는다
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def record(self, name: str, **kwargs) -> None:
        self.calls.append((name, kwargs))

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


class _Memory:
    def __init__(self, rec: _Recorder, result: str = "[관련 기억]\n- 사용자는 대학생"):
        self._rec = rec
        self._result = result

    def to_prompt(self, query: str, top_k: int = 5, token_budget: int = 0) -> str:
        self._rec.record("memory.to_prompt", query=query, top_k=top_k)
        return self._result


class _Knowledge:
    def __init__(self, rec: _Recorder, result: str = "[지식]\n활빈당은 의적 집단이다"):
        self._rec = rec
        self._result = result

    def to_index(self) -> str:
        return "[찾아볼 수 있는 것]\n- 인물: 활빈당 두목"

    def base_text(self) -> str:
        return "[배경 지식]\n조선은 신분제 사회다"

    def search_relevant(self, query: str, token_budget: int = 500) -> str:
        self._rec.record("knowledge.search_relevant", query=query)
        return self._result

    def get_relationships_for(self, character: str) -> list[dict]:
        self._rec.record("knowledge.get_relationships_for", character=character)
        return [{"from": "홍길동", "to": character, "type": "동료", "sentiment": "신뢰"}]


class _FewShot:
    def __init__(self, rec: _Recorder, result: str = "[예시 대화]\n사용자: 안녕\n캐릭터: 반갑네"):
        self._rec = rec
        self._result = result

    def to_prompt(self, query: str, emotions=None, top_k: int = 3, token_budget: int = 300) -> str:
        self._rec.record("fewshot.to_prompt", query=query, emotions=emotions)
        return self._result


class _History:
    def __init__(self, rec: _Recorder, result: str = "[최근 대화]\n사용자: 안녕"):
        self._rec = rec
        self._result = result

    def to_prompt(self, n: int = 10) -> str:
        self._rec.record("history.to_prompt", n=n)
        return self._result


class _Emotion:
    def __init__(self, rec: _Recorder, result: str = "[현재 감정 상태]\n- 장난기: 0.700"):
        self._rec = rec
        self._result = result

    def to_prompt(self) -> str:
        self._rec.record("emotion.to_prompt")
        return self._result

    def get_state(self) -> dict[str, float]:
        return {"장난기": 0.7}


class _Persona:
    def to_system_prompt(self) -> str:
        return "[페르소나]\n이름: 홍길동"

    def get_behavior_section(self) -> str:
        return "[행동 지침]\n- 약자 앞에서는 물러서지 않는다"

    def get_inner_world(self) -> str:
        return "[내면 상태]\n- 아버지를 향한 응어리"

    def get_relationships(self) -> list[dict]:
        return [{"target": "임꺽정", "type": "친구", "description": "오랜 벗"}]


class _WorkingMemory:
    def __init__(self, prompt: str = ""):
        self._prompt = prompt

    def to_prompt(self) -> str:
        return self._prompt


class _ToolCallsOnly:
    """기본 상태 조립 호출을 기록에서 걷어낸다.

    뇌는 첫 LLM 호출 **전에** 감정·최근 대화·지식 목차를 읽는다. 그것까지
    기록에 남으면 "도구가 모듈을 정확히 불렀는가"를 볼 수 없다.
    """

    def __init__(self, client, rec: _Recorder):
        self._client = client
        self._rec = rec
        self._cleared = False

    def call_llm(self, **kwargs):
        if not self._cleared:
            self._rec.calls.clear()
            self._cleared = True
        return self._client.call_llm(**kwargs)


def _brain(client, *, rec: _Recorder | None = None, working_memory=None, max_iterations: int = 5):
    rec = rec or _Recorder()
    registry = ToolRegistry(
        persona=_Persona(),
        emotion=_Emotion(rec),
        memory=_Memory(rec),
        knowledge=_Knowledge(rec),
        history=_History(rec),
        fewshot=_FewShot(rec),
    )
    return (
        ReActBrain(
            client=_ToolCallsOnly(client, rec),
            persona=_Persona(),
            tools=registry,
            working_memory=working_memory or _WorkingMemory(),
            max_iterations=max_iterations,
        ),
        rec,
    )


_FINISH = dict(situation="사용자가 인사했다", intent="반갑게 받는다", avoid="과장", tone="친근")


# ---------------------------------------------------------------------------
# 1. 루프 제어 (REQ-RA-01 ~ 07)
# ---------------------------------------------------------------------------


class TestLoopControl:
    def test_immediate_finish_takes_one_iteration(self):
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client)

        bundle = brain.think("안녕")

        assert bundle.iterations == 1
        assert bundle.hit_cap is False
        assert bundle.strategy.situation == "사용자가 인사했다"

    def test_multi_step_reasoning(self):
        """도구 결과를 보고 다시 파고드는 재귀가 가능해야 한다."""
        client = ScriptedClient(
            [
                tool_step(tool_call("search_memory", query="시험")),
                tool_step(tool_call("search_knowledge", query="대학 시험")),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, rec = _brain(client)

        bundle = brain.think("시험 망했어")

        assert bundle.iterations == 3
        assert rec.names() == ["memory.to_prompt", "knowledge.search_relevant"]

    def test_parallel_tool_calls_all_execute(self):
        client = ScriptedClient(
            [
                tool_step(
                    tool_call("search_memory", query="시험"),
                    tool_call("get_history", n=5),
                ),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, rec = _brain(client)

        bundle = brain.think("시험 망했어")

        assert rec.names() == ["memory.to_prompt", "history.to_prompt"]
        assert "search_memory" in bundle.collected
        assert "get_history" in bundle.collected

    def test_cap_stops_loop_and_marks_hit_cap(self):
        client = ScriptedClient(
            [tool_step(tool_call("search_memory", query=f"q{i}")) for i in range(10)]
        )
        brain, _ = _brain(client, max_iterations=5)

        bundle = brain.think("복잡한 질문")

        assert bundle.iterations == 5
        assert bundle.hit_cap is True
        assert client.call_count == 5

    def test_cap_is_not_a_failure(self):
        """상한 도달은 정상 종료다 (REQ-RA-42). 수집분은 살아 있어야 한다."""
        client = ScriptedClient(
            [tool_step(tool_call("search_memory", query=f"q{i}")) for i in range(10)]
        )
        brain, _ = _brain(client, max_iterations=5)

        bundle = brain.think("복잡한 질문")

        assert bundle.collected.get("search_memory")

    def test_duplicate_call_is_reported_as_observation(self):
        client = ScriptedClient(
            [
                tool_step(tool_call("search_memory", query="시험")),
                tool_step(tool_call("search_memory", query="시험")),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, _ = _brain(client)

        brain.think("시험")

        third_call_messages = client.calls[2]["messages"]
        observations = [m["content"] for m in third_call_messages if m.get("role") == "tool"]
        assert any("이미" in o for o in observations)

    def test_llm_call_count_never_exceeds_cap(self):
        client = ScriptedClient([tool_step(tool_call("get_history", n=i + 1)) for i in range(20)])
        brain, _ = _brain(client, max_iterations=3)

        brain.think("안녕")

        assert client.call_count == 3


# ---------------------------------------------------------------------------
# 2. 도구 (REQ-RA-10 ~ 19)
# ---------------------------------------------------------------------------


class TestTools:
    def test_all_tools_are_advertised(self):
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client)

        brain.think("안녕")

        names = {t["function"]["name"] for t in client.calls[0]["tools"]}
        assert names == {
            "search_memory",
            "search_knowledge",
            "search_fewshot",
            "get_history",
            FINISH_TOOL,
        }

    def test_search_memory_passes_rewritten_query(self):
        """뇌가 재작성한 쿼리가 그대로 모듈에 전달된다 — 사용자 입력 원문이 아니다."""
        client = ScriptedClient(
            [
                tool_step(tool_call("search_memory", query="사용자의 학업 상황", top_k=3)),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, rec = _brain(client)

        brain.think("시험 망했어 ㅠㅠ")

        assert rec.calls[0] == ("memory.to_prompt", {"query": "사용자의 학업 상황", "top_k": 3})

    def test_search_knowledge_calls_module(self):
        client = ScriptedClient(
            [
                tool_step(tool_call("search_knowledge", query="활빈당")),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, rec = _brain(client)

        brain.think("활빈당이 뭐야")

        assert rec.calls[0] == ("knowledge.search_relevant", {"query": "활빈당"})

    def test_search_fewshot_passes_emotion_when_requested(self):
        client = ScriptedClient(
            [
                tool_step(tool_call("search_fewshot", query="위로", use_emotion=True)),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, rec = _brain(client)

        brain.think("힘들어")

        assert rec.calls[0][1]["emotions"] == {"장난기": 0.7}

    def test_search_fewshot_omits_emotion_when_disabled(self):
        client = ScriptedClient(
            [
                tool_step(tool_call("search_fewshot", query="위로", use_emotion=False)),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, rec = _brain(client)

        brain.think("힘들어")

        assert rec.calls[0][1]["emotions"] is None

    def test_get_history_passes_n(self):
        client = ScriptedClient(
            [tool_step(tool_call("get_history", n=3)), tool_step(finish_call(**_FINISH))]
        )
        brain, rec = _brain(client)

        brain.think("아까 뭐라고 했지")

        assert rec.calls[0] == ("history.to_prompt", {"n": 3})

    def test_bad_arguments_are_returned_as_observation(self):
        client = ScriptedClient(
            [
                tool_step(tool_call("search_memory")),  # 필수 인자 query 누락
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, rec = _brain(client)

        bundle = brain.think("안녕")

        assert bundle.iterations == 2
        assert rec.calls == []
        observations = [
            m["content"] for m in client.calls[1]["messages"] if m.get("role") == "tool"
        ]
        assert any("query" in o for o in observations)

    def test_unknown_tool_returns_tool_list(self):
        client = ScriptedClient(
            [
                tool_step(tool_call("search_internet", query="날씨")),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, _ = _brain(client)

        brain.think("날씨 어때")

        observations = [
            m["content"] for m in client.calls[1]["messages"] if m.get("role") == "tool"
        ]
        assert any("search_memory" in o for o in observations)

    def test_tool_failure_does_not_kill_the_loop(self):
        """도구 하나가 죽어도 뇌는 다른 경로를 택할 수 있어야 한다 (REQ-RA-43)."""

        class _BrokenMemory:
            def to_prompt(self, query: str, top_k: int = 5, token_budget: int = 0) -> str:
                raise RuntimeError("임베딩 모델 없음")

        rec = _Recorder()
        registry = ToolRegistry(
            persona=_Persona(),
            emotion=_Emotion(rec),
            memory=_BrokenMemory(),
            knowledge=_Knowledge(rec),
            history=_History(rec),
            fewshot=_FewShot(rec),
        )
        client = ScriptedClient(
            [
                tool_step(tool_call("search_memory", query="시험")),
                tool_step(tool_call("search_knowledge", query="시험")),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain = ReActBrain(
            client=client,
            persona=_Persona(),
            tools=registry,
            working_memory=_WorkingMemory(),
        )

        bundle = brain.think("시험")

        assert bundle.iterations == 3
        assert "search_knowledge" in bundle.collected
        assert bundle.tool_calls[0].error


# ---------------------------------------------------------------------------
# 3. 산출물 (REQ-RA-20 ~ 24)
# ---------------------------------------------------------------------------


class TestBundle:
    def test_uncalled_sources_stay_empty(self):
        client = ScriptedClient(
            [tool_step(tool_call("get_history", n=3)), tool_step(finish_call(**_FINISH))]
        )
        brain, _ = _brain(client)

        bundle = brain.think("안녕")

        assert set(bundle.collected) == {"get_history"}

    def test_strategy_carries_four_elements(self):
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client)

        strategy = brain.think("안녕").strategy

        assert strategy.situation == "사용자가 인사했다"
        assert strategy.intent == "반갑게 받는다"
        assert strategy.avoid == "과장"
        assert strategy.tone == "친근"

    def test_repeated_tool_results_accumulate(self):
        client = ScriptedClient(
            [
                tool_step(tool_call("search_memory", query="시험")),
                tool_step(tool_call("search_memory", query="학교")),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, _ = _brain(client)

        bundle = brain.think("시험")

        assert bundle.collected["search_memory"].count("사용자는 대학생") == 2

    def test_tool_calls_are_recorded_for_trace(self):
        client = ScriptedClient(
            [
                tool_step(tool_call("search_memory", query="시험")),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, _ = _brain(client)

        bundle = brain.think("시험")

        record = bundle.tool_calls[0]
        assert record.iteration == 1
        assert record.name == "search_memory"
        # 기본값까지 채워진 최종 인자가 남는다 — 실제로 그렇게 호출했기 때문이다.
        assert record.arguments == {"query": "시험", "top_k": 5}
        assert record.result_len > 0

    def test_working_memory_updates_are_carried(self):
        client = ScriptedClient(
            [
                tool_step(
                    finish_call(
                        **_FINISH,
                        resolved=["abc123"],
                        new_thoughts=[
                            {"kind": "question", "content": "왜 말수가 줄었을까"},
                        ],
                    )
                )
            ]
        )
        brain, _ = _brain(client)

        bundle = brain.think("...")

        assert bundle.resolved_ids == ["abc123"]
        assert bundle.new_thoughts[0].content == "왜 말수가 줄었을까"

    def test_cap_yields_empty_strategy(self):
        """상한에 걸리면 전략이 없다. 없는 것을 지어내지 않는다."""
        client = ScriptedClient(
            [tool_step(tool_call("get_history", n=i + 1)) for i in range(6)],
        )
        brain, _ = _brain(client, max_iterations=2)

        bundle = brain.think("안녕")

        assert bundle.strategy.situation == ""
        assert bundle.strategy.intent == ""


# ---------------------------------------------------------------------------
# 4. 작업기억 주입 (REQ-RA-53)
# ---------------------------------------------------------------------------


class TestWorkingMemoryInjection:
    def test_working_memory_appears_in_system_prompt(self):
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(
            client,
            working_memory=_WorkingMemory("[미해결 사고]\n- (q:abc123) 왜 말수가 줄었을까"),
        )

        brain.think("안녕")

        system = client.calls[0]["messages"][0]["content"]
        assert "왜 말수가 줄었을까" in system

    def test_persona_appears_in_system_prompt(self):
        """뇌도 자기가 누구인지 알아야 전략을 세울 수 있다."""
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client)

        brain.think("안녕")

        assert "홍길동" in client.calls[0]["messages"][0]["content"]

    def test_user_input_is_passed(self):
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client)

        brain.think("시험 망했어")

        assert any("시험 망했어" in str(m.get("content")) for m in client.calls[0]["messages"])


# ---------------------------------------------------------------------------
# 5. 실패 (REQ-RA-40 ~ 44)
# ---------------------------------------------------------------------------


class TestFailure:
    def test_llm_error_raises_classified_error(self):
        client = ScriptedClient([RuntimeError("연결 끊김")])
        brain, _ = _brain(client)

        with pytest.raises(BrainError) as exc:
            brain.think("안녕")

        assert exc.value.reason_code == "llm_error"

    def test_provider_refusal_raises_classified_error(self):
        client = ScriptedClient([text_step("Your request was rejected by the provider")])
        brain, _ = _brain(client)

        with pytest.raises(BrainError) as exc:
            brain.think("안녕")

        assert exc.value.reason_code == "refusal"

    def test_unparseable_tool_arguments_fail_after_retry(self):
        client = ScriptedClient(
            [
                tool_step(tool_call("search_memory", raw_arguments="{망가진")),
                tool_step(tool_call("search_memory", raw_arguments="{또 망가진")),
            ]
        )
        brain, _ = _brain(client)

        with pytest.raises(BrainError) as exc:
            brain.think("안녕")

        assert exc.value.reason_code == "bad_tool_args"

    def test_finish_schema_violation_fails(self):
        """부분 해석해서 진행하지 않는다 (REQ-RA-24)."""
        client = ScriptedClient(
            [tool_step(tool_call(FINISH_TOOL, raw_arguments=json.dumps({"situation": "인사"})))]
        )
        brain, _ = _brain(client)

        with pytest.raises(BrainError) as exc:
            brain.think("안녕")

        assert exc.value.reason_code == "bad_finish"

    def test_unparseable_finish_fails(self):
        client = ScriptedClient([tool_step(tool_call(FINISH_TOOL, raw_arguments="{깨짐"))])
        brain, _ = _brain(client)

        with pytest.raises(BrainError) as exc:
            brain.think("안녕")

        assert exc.value.reason_code == "bad_finish"

    def test_no_fallback_collection_on_failure(self):
        """실패는 실패다. 고정 규칙 수집으로 몰래 대체하지 않는다 (REQ-RA-41)."""
        client = ScriptedClient([RuntimeError("연결 끊김")])
        brain, rec = _brain(client)

        with pytest.raises(BrainError):
            brain.think("안녕")

        assert rec.calls == []

    def test_response_without_tool_calls_is_nudged(self):
        """도구도 finish도 없는 응답은 실패가 아니라 재촉 대상이다."""
        client = ScriptedClient([text_step("음... 생각 중"), tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client)

        bundle = brain.think("안녕")

        assert bundle.iterations == 2


# ---------------------------------------------------------------------------
# 6. 기본 상태 (REQ-RA-70 ~ 75)
#
# 뇌가 자기 상태를 모르는 채로 판단을 시작하면, 무엇을 검색해야 할지 자체를
# 정할 수 없다. 항상 참인 것은 프롬프트에, 찾아야 아는 것만 도구에 둔다.
# ---------------------------------------------------------------------------


class TestBaselineState:
    def _system_prompt(self, **brain_kwargs) -> str:
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client, **brain_kwargs)
        brain.think("안녕")
        return client.calls[0]["messages"][0]["content"]

    def test_current_emotion_is_known_without_a_tool(self):
        assert "장난기" in self._system_prompt()

    def test_behavior_guide_is_known(self):
        assert "행동 지침" in self._system_prompt()

    def test_inner_world_is_known(self):
        assert "내면" in self._system_prompt()

    def test_recent_history_is_known(self):
        """지시어("그 사람", "아까 그거")를 해석하려면 최근 대화가 필요하다."""
        assert "최근 대화" in self._system_prompt()

    def test_knowledge_index_is_known(self):
        assert "활빈당" in self._system_prompt()

    def test_background_knowledge_is_known(self):
        """배경지식은 검색 없이 이미 알고 있어야 한다 (TASK-20)."""
        assert "신분제" in self._system_prompt()

    def test_knowledge_body_is_not_inlined(self):
        """목차만 싣는다. 본문까지 실으면 루프마다 재전송된다 (REQ-RA-72)."""
        assert "의적 집단이다" not in self._system_prompt()

    def test_emotion_tool_is_gone(self):
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client)

        brain.think("안녕")

        names = {t["function"]["name"] for t in client.calls[0]["tools"]}
        assert "get_emotion" not in names

    def test_history_tool_remains(self):
        """기본 5턴보다 더 거슬러 올라갈 수단은 남긴다 (REQ-RA-75)."""
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client)

        brain.think("안녕")

        names = {t["function"]["name"] for t in client.calls[0]["tools"]}
        assert "get_history" in names


class TestBaselineReachesStageTwo:
    def _bundle(self):
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client)
        return brain.think("안녕")

    def test_emotion_is_carried_in_bundle(self):
        """뇌가 본 감정과 발화가 보는 감정이 어긋나면 안 된다 (REQ-RA-73)."""
        assert "장난기" in self._bundle().baseline["emotion"]

    def test_history_is_carried_in_bundle(self):
        assert self._bundle().baseline["history"]

    def test_background_knowledge_is_carried(self):
        """배경지식은 캐릭터의 상식이다. 말할 때도 알고 있어야 한다."""
        assert "신분제" in self._bundle().baseline["knowledge"]

    def test_index_is_not_carried(self):
        """목차는 '무엇을 더 찾아볼 수 있는가'다. 발화 재료가 아니다 (REQ-RA-74)."""
        assert "찾아볼 수 있는 것" not in self._bundle().baseline.get("knowledge", "")


# ---------------------------------------------------------------------------
# 7. 절제 (REQ-RA-81 ~ 83)
#
# 지연의 60~70%가 뇌였고, 원인은 루프 수가 아니라 출력 길이였다.
# 출력 토큰은 곧 시간이다.
# ---------------------------------------------------------------------------


class TestRestraint:
    def test_long_strategy_field_is_truncated(self):
        long_text = "가" * 500
        client = ScriptedClient([tool_step(finish_call(**{**_FINISH, "situation": long_text}))])
        brain, _ = _brain(client)

        strategy = brain.think("안녕").strategy

        assert len(strategy.situation) <= MAX_STRATEGY_CHARS

    def test_short_strategy_field_is_untouched(self):
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client)

        assert brain.think("안녕").strategy.situation == "사용자가 인사했다"

    def test_long_thought_is_truncated_before_accumulating(self):
        """사고문이 그대로 쌓이면 4루프에서 입력이 1만 토큰을 넘는다."""
        client = ScriptedClient(
            [
                TrimmedMessage(
                    content="나" * 2000,
                    role="assistant",
                    reasoning_content="",
                    tool_calls=[tool_call("search_memory", query="시험")],
                    usage=None,
                ),
                tool_step(finish_call(**_FINISH)),
            ]
        )
        brain, _ = _brain(client)

        brain.think("시험")

        accumulated = next(m for m in client.calls[1]["messages"] if m.get("role") == "assistant")
        assert len(accumulated["content"]) <= MAX_THOUGHT_CHARS

    def test_output_cap_is_sent_to_the_provider(self):
        """프롬프트 지시만으로는 폭주가 막히지 않는다 (REQ-RA-83)."""
        client = ScriptedClient([tool_step(finish_call(**_FINISH))])
        brain, _ = _brain(client)

        brain.think("안녕")

        assert client.calls[0]["max_tokens"] == BRAIN_MAX_OUTPUT_TOKENS
