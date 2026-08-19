"""ReAct 뇌 — 캐릭터의 생각을 담당한다 (SPEC-09).

이전 Stage 1은 모든 모듈을 고정 순서로 한 번씩 부르고 그 결과를 버렸다.
뇌는 대신 스스로 묻는다. *지금 이 말에 답하려면 무엇을 알아야 하는가.*
도구로 확인하고, 결과를 보고 다시 파고들고, 충분하다고 판단하면 방침을 세운다.

실패를 감추지 않는다 (REQ-RA-40 · 41). 생각이 죽으면 턴이 죽는다.
고정 규칙 수집으로 몰래 갈아타면 품질 저하가 원인 불명으로 남는다.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from src.agent.schemas import NewThought, ResponseStrategy, ThoughtBundle, ToolCallRecord
from src.agent.tools import FINISH_TOOL, ToolArgumentError, ToolRegistry, UnknownToolError
from src.validity import provider_error_reason

MAX_ITERATIONS = 5

# 인자를 잘못 만든 호출에 주는 기회. 한 번은 고쳐 쓸 수 있게 하되,
# 두 번째도 같으면 모델이 스키마를 못 읽는 것이므로 턴을 세운다.
MAX_BAD_ARGUMENT_RETRIES = 2

# 출력 절제 (SPEC-09 REQ-RA-81 ~ 83).
#
# 실사용에서 턴 지연의 60~70%가 뇌였고, 원인은 루프 수가 아니라 출력 길이였다.
# 전략 4필드에 각각 세 줄씩 쓰면서 턴당 1,961 출력 토큰을 썼다. 사고문까지
# 그대로 누적되어 4루프째 입력이 13,842 토큰에 달했다.
#
# 프롬프트로 "짧게"를 지시하되, 지켜지지 않을 때를 대비해 세 겹으로 막는다.
MAX_STRATEGY_CHARS = 200
MAX_THOUGHT_CHARS = 300
BRAIN_MAX_OUTPUT_TOKENS = 800

_SYSTEM_TEMPLATE = """너는 아래 인물의 '생각'이다. 아직 말하지 않는다.
사용자의 말에 답하기 전에, 무엇을 알아야 하는지 스스로 판단하고 도구로 확인한 뒤
어떻게 반응할지 방침을 정하는 것이 네 일이다.

{persona}
{state}
[생각하는 방법]
- 생각은 짧게 쓴다. 도구를 부를 때 이유를 길게 적지 않는다 — 두 문장이면 충분하다.
- 위에 이미 있는 것(성격·행동 지침·내면·감정·최근 대화)은 다시 찾지 않는다.
- 사용자의 말에서 확인이 필요한 것을 정하고 도구를 부른다.
- [내가 아는 것]에 있는 항목이 대화와 얽히면 search_knowledge로 본문을 꺼낸다.
- 도구 결과를 보고 더 파고들 것이 있으면 다시 부른다. 결과에 없는 것을 지어내지 않는다.
- 검색어는 사용자 말 그대로가 아니라 '무엇을 떠올리려는지'로 적는다.
- 확인할 것이 없으면 곧바로 {finish}를 불러도 된다.
- 충분하다고 판단되면 {finish}로 방침(상황·할 말·피할 것·태도)을 확정한다.
  네 항목은 **각각 한 문장**으로 적는다. 길게 쓰면 잘린다.
- 최대 {max_iterations}번까지 생각할 수 있다.
- {finish}의 new_thoughts에는 아직 답을 얻지 못한 질문이나 이 사람에 대한 추측만 남긴다.
  확정된 사실은 남기지 않는다 — 그것은 기억이 맡는다.
- 이전에 남긴 미해결 사고가 이번 대화로 풀렸다면 그 id를 resolved에 넣는다."""

_NUDGE = f"도구를 부르거나 {FINISH_TOOL}로 생각을 마쳐라."


class BrainError(RuntimeError):
    """뇌가 생각을 마치지 못했다. `reason_code`로 원인을 분류한다 (REQ-RA-44)."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = reason_code


class ReActBrain:
    """관찰 → 사고 → 도구 호출을 반복하며 응답 전략을 세운다."""

    def __init__(
        self,
        client,
        persona,
        tools: ToolRegistry,
        working_memory,
        max_iterations: int = MAX_ITERATIONS,
        log: Callable[[str], None] | None = None,
    ):
        self._client = client
        self._persona = persona
        self._tools = tools
        self._working_memory = working_memory
        self._max_iterations = max_iterations
        self._log = log or (lambda _msg: None)

    # ─── 진입점 ───

    def think(self, user_input: str) -> ThoughtBundle:
        # 기본 상태는 한 번만 읽는다. 같은 값이 뇌의 프롬프트와 번들 양쪽에 쓰이므로,
        # 두 번 읽으면 그 사이 상태가 바뀌었을 때 뇌와 발화가 다른 것을 본다.
        baseline = self._tools.baseline()

        messages = [
            {"role": "system", "content": self._system_prompt(baseline)},
            {"role": "user", "content": user_input},
        ]

        collected: dict[str, str] = {}
        records: list[ToolCallRecord] = []
        seen: set[tuple[str, str]] = set()
        bad_arguments = 0
        iteration = 0

        while iteration < self._max_iterations:
            iteration += 1
            result = self._call(messages)

            if not result.tool_calls:
                reason = provider_error_reason(result.content)
                if reason:
                    raise BrainError("refusal", reason)
                self._log(f"[{iteration}] 도구 없이 응답 — 재촉")
                messages.append({"role": "assistant", "content": result.content})
                messages.append({"role": "user", "content": _NUDGE})
                continue

            messages.append(self._assistant_message(result))

            for call in result.tool_calls:
                arguments, error = self._parse_arguments(call)
                if error is not None:
                    if self._tools.is_finish(call.name):
                        raise BrainError("bad_finish", error)
                    bad_arguments += 1
                    if bad_arguments >= MAX_BAD_ARGUMENT_RETRIES:
                        raise BrainError("bad_tool_args", error)
                    messages.append(self._observation(call, error))
                    continue

                if self._tools.is_finish(call.name):
                    return self._finish(arguments, baseline, collected, records, iteration)

                observation = self._invoke(call, arguments, iteration, collected, records, seen)
                messages.append(self._observation(call, observation))

        self._log(f"상한 {self._max_iterations}회 도달 — 수집분으로 마감")
        return ThoughtBundle(
            strategy=ResponseStrategy(),
            baseline=baseline,
            collected=collected,
            iterations=iteration,
            hit_cap=True,
            tool_calls=records,
        )

    # ─── LLM ───

    def _call(self, messages: list[dict]):
        try:
            return self._client.call_llm(
                messages=messages,
                tools=self._tools.specs(),
                use_stream=False,
                mute=True,
                max_tokens=BRAIN_MAX_OUTPUT_TOKENS,
            )
        except Exception as e:
            raise BrainError("llm_error", f"{type(e).__name__}: {e}") from e

    def _system_prompt(self, baseline: dict[str, str]) -> str:
        """도구를 부르기 전에 이미 아는 것을 모두 싣는다 (REQ-RA-70).

        자기 감정도 모르고 무엇을 아는지도 모르는 채로 판단을 시작하면,
        검색할 대상을 정하는 일 자체가 찍기가 된다.
        """
        sections = [
            self._persona.get_behavior_section(),
            self._persona.get_inner_world(),
            baseline.get("knowledge", ""),
            baseline.get("emotion", ""),
            baseline.get("history", ""),
            self._tools.knowledge_index(),
            self._working_memory.to_prompt(),
        ]
        state = "\n\n".join(s for s in sections if s)

        return _SYSTEM_TEMPLATE.format(
            persona=self._persona.to_system_prompt(),
            state=f"\n{state}\n" if state else "",
            finish=FINISH_TOOL,
            max_iterations=self._max_iterations,
        )

    @staticmethod
    def _assistant_message(result) -> dict:
        """사고문은 잘라서 누적한다. 그대로 쌓으면 루프마다 입력이 불어난다."""
        return {
            "role": "assistant",
            "content": (result.content or "")[:MAX_THOUGHT_CHARS],
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in result.tool_calls
            ],
        }

    @staticmethod
    def _observation(call, content: str) -> dict:
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    # ─── 도구 ───

    def _parse_arguments(self, call) -> tuple[dict, str | None]:
        """인자를 파싱·검증한다. 실패하면 (빈 dict, 사유)를 돌려준다."""
        raw = call.arguments or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            return {}, f"인자를 JSON으로 읽을 수 없다: {e}"

        if not isinstance(parsed, dict):
            return {}, "인자는 객체여야 한다"

        try:
            return self._tools.validate(call.name, parsed), None
        except UnknownToolError:
            return {}, f"'{call.name}'은 없는 도구다. 사용 가능: {', '.join(self._tools.names())}"
        except ToolArgumentError as e:
            return {}, str(e)

    def _invoke(
        self,
        call,
        arguments: dict,
        iteration: int,
        collected: dict[str, str],
        records: list[ToolCallRecord],
        seen: set[tuple[str, str]],
    ) -> str:
        signature = (call.name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
        if signature in seen:
            self._log(f"[{iteration}] {call.name} 중복 호출")
            return "이미 같은 조건으로 조회했다. 다른 각도로 찾거나 생각을 마쳐라."
        seen.add(signature)

        started = time.perf_counter()
        try:
            observation = self._tools.execute(call.name, arguments)
        except Exception as e:
            # 도구 하나가 죽어도 루프는 계속된다. 뇌가 다른 경로를 택할 수 있어야 한다.
            message = f"{type(e).__name__}: {e}"
            self._log(f"[{iteration}] {call.name} 실패 — {message}")
            records.append(
                ToolCallRecord(
                    iteration=iteration,
                    name=call.name,
                    arguments=arguments,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error=message,
                )
            )
            return f"도구 실행 실패: {message}"

        self._log(f"[{iteration}] {call.name}({arguments}) → {len(observation)}자")
        records.append(
            ToolCallRecord(
                iteration=iteration,
                name=call.name,
                arguments=arguments,
                result_len=len(observation),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        )
        collected[call.name] = (
            f"{collected[call.name]}\n\n{observation}" if call.name in collected else observation
        )
        return observation

    # ─── 종료 ───

    def _finish(
        self,
        arguments: dict,
        baseline: dict[str, str],
        collected: dict[str, str],
        records: list[ToolCallRecord],
        iteration: int,
    ) -> ThoughtBundle:
        strategy = ResponseStrategy(
            situation=self._clip("situation", arguments["situation"]),
            intent=self._clip("intent", arguments["intent"]),
            avoid=self._clip("avoid", arguments.get("avoid", "")),
            tone=self._clip("tone", arguments["tone"]),
        )
        self._log(f"생각 종료 ({iteration}회) — {strategy.situation}")

        return ThoughtBundle(
            strategy=strategy,
            baseline=baseline,
            collected=collected,
            iterations=iteration,
            hit_cap=False,
            tool_calls=records,
            resolved_ids=[str(i) for i in (arguments.get("resolved") or [])],
            new_thoughts=self._parse_thoughts(arguments.get("new_thoughts") or []),
        )

    def _clip(self, field: str, value: str) -> str:
        """전략 항목을 한 문장 분량으로 줄인다. 자른 사실은 감추지 않는다."""
        if len(value) <= MAX_STRATEGY_CHARS:
            return value
        self._log(f"전략 '{field}' {len(value)}자 → {MAX_STRATEGY_CHARS}자로 자름")
        return value[:MAX_STRATEGY_CHARS]

    @staticmethod
    def _parse_thoughts(raw: list) -> list[NewThought]:
        thoughts = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("kind") or not item.get("content"):
                raise BrainError("bad_finish", f"new_thoughts 항목이 형식에 맞지 않는다: {item}")
            thoughts.append(
                NewThought(
                    kind=str(item["kind"]),
                    content=str(item["content"]),
                    confidence=float(item.get("confidence") or 0.0),
                )
            )
        return thoughts
