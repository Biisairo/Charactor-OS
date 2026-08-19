"""뇌가 쓰는 도구 (SPEC-09 REQ-RA-10 ~ 19).

도구는 기존 모듈 API에 1:1로 대응한다. 새 검색 로직을 여기에 만들지 않는다 —
검색이 두 곳에 정의되면 반드시 어긋나고, 그것이 이 작업의 출발점이었다.

모든 도구는 read-only다. 상태를 바꾸는 일은 Stage 3의 몫이다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

FINISH_TOOL = "finish"

# 도구 단계에서는 넉넉히 가져오고, 최종 예산은 PromptEngine이 자른다.
# 여기서 미리 조이면 뇌가 판단할 재료까지 잘려나간다.
TOOL_TOKEN_BUDGET = 1500

# 기본으로 쥐고 시작하는 대화 이력. 지시어("그 사람", "아까 그거")를 풀려면
# 이만큼은 도구 없이 보여야 한다. 더 거슬러 올라갈 때 get_history를 쓴다.
BASELINE_HISTORY_TURNS = 5


class UnknownToolError(Exception):
    """등록되지 않은 도구 이름."""


class ToolArgumentError(Exception):
    """도구 인자가 스키마를 만족하지 않는다."""


@dataclass(frozen=True)
class _Param:
    name: str
    type: str
    description: str
    required: bool = False
    default: object = None


@dataclass(frozen=True)
class _Tool:
    name: str
    description: str
    params: tuple[_Param, ...]
    handler: Callable[..., str] | None = None


_PYTHON_TYPES = {"string": str, "integer": int, "boolean": bool}


class ToolRegistry:
    """도구 정의와 모듈 바인딩을 한곳에 모은다."""

    def __init__(self, persona, emotion, memory, knowledge, history, fewshot):
        # persona는 관계를 시스템 프롬프트에 직접 싣는다. 관계 조회 도구는
        # 폐지했고 `search_knowledge`가 그 자리를 대신한다 (TASK-22).
        self._persona = persona
        self._emotion = emotion
        self._memory = memory
        self._knowledge = knowledge
        self._history = history
        self._fewshot = fewshot
        self._tools = {t.name: t for t in self._build()}

    # ─── 정의 ───

    def _build(self) -> list[_Tool]:
        return [
            _Tool(
                name="search_memory",
                description=(
                    "이 사람과의 대화에서 쌓인 기억을 찾는다. "
                    "질문 원문이 아니라 '무엇을 떠올리려는지'를 쿼리로 적어라."
                ),
                params=(
                    _Param("query", "string", "떠올리려는 것", required=True),
                    _Param("top_k", "integer", "최대 개수", default=5),
                ),
                handler=self._search_memory,
            ),
            _Tool(
                name="search_knowledge",
                description="세계관·인물·장소 등 내가 아는 설정을 찾는다.",
                params=(_Param("query", "string", "알아보려는 것", required=True),),
                handler=self._search_knowledge,
            ),
            _Tool(
                name="search_fewshot",
                description="비슷한 상황에서 내가 어떻게 말했는지 예시를 찾는다.",
                params=(
                    _Param("query", "string", "상황 묘사", required=True),
                    _Param("use_emotion", "boolean", "지금 감정을 반영할지", default=True),
                ),
                handler=self._search_fewshot,
            ),
            _Tool(
                name="get_history",
                description="최근 대화 흐름을 되짚는다.",
                params=(_Param("n", "integer", "가져올 턴 수", default=10),),
                handler=self._get_history,
            ),
            _Tool(
                name=FINISH_TOOL,
                description=(
                    "생각을 마치고 응답 방침을 확정한다. 더 찾을 것이 없다고 판단하면 호출한다."
                ),
                params=(
                    _Param("situation", "string", "지금 무슨 상황인가", required=True),
                    _Param("intent", "string", "무엇을 말할 것인가", required=True),
                    _Param("tone", "string", "어떤 태도로 말할 것인가", required=True),
                    _Param("avoid", "string", "말하지 않을 것", default=""),
                    _Param("resolved", "array", "해소된 미해결 사고의 id 목록", default=None),
                    _Param("new_thoughts", "array", "새로 남길 미해결 질문·가설", default=None),
                ),
            ),
        ]

    # ─── 조회 ───

    def specs(self) -> list[dict]:
        """LLM에 넘길 도구 정의 (OpenAI tools 형식)."""
        return [self._spec(tool) for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def _spec(self, tool: _Tool) -> dict:
        properties = {}
        required = []
        for param in tool.params:
            schema: dict = {"description": param.description}
            if param.type == "array":
                schema["type"] = "array"
                schema["items"] = (
                    {"type": "string"}
                    if param.name == "resolved"
                    else {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["question", "hypothesis"]},
                            "content": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["kind", "content"],
                    }
                )
            else:
                schema["type"] = param.type
            properties[param.name] = schema
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    # ─── 실행 ───

    def validate(self, name: str, arguments: dict) -> dict:
        """인자를 검증하고 기본값을 채운 사본을 돌려준다.

        위반은 예외로 알린다. 뇌가 그것을 관찰로 바꿔 LLM에 되돌려준다 —
        틀린 호출을 조용히 보정하면 다음에도 같은 실수를 반복한다.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(name)

        resolved = {}
        for param in tool.params:
            if param.name not in arguments or arguments[param.name] is None:
                if param.required:
                    raise ToolArgumentError(f"필수 인자 '{param.name}'이 없다")
                if param.default is not None:
                    resolved[param.name] = param.default
                continue

            value = arguments[param.name]
            expected = _PYTHON_TYPES.get(param.type)
            if expected is bool and not isinstance(value, bool):
                raise ToolArgumentError(f"인자 '{param.name}'은 true/false여야 한다")
            if expected is int and isinstance(value, bool):
                raise ToolArgumentError(f"인자 '{param.name}'은 정수여야 한다")
            if expected is int and not isinstance(value, int):
                raise ToolArgumentError(f"인자 '{param.name}'은 정수여야 한다")
            if expected is str and not isinstance(value, str):
                raise ToolArgumentError(f"인자 '{param.name}'은 문자열이어야 한다")
            if param.type == "array" and not isinstance(value, list):
                raise ToolArgumentError(f"인자 '{param.name}'은 배열이어야 한다")
            resolved[param.name] = value

        return resolved

    def execute(self, name: str, arguments: dict) -> str:
        """도구를 실행하고 관찰 텍스트를 돌려준다."""
        tool = self._tools.get(name)
        if tool is None or tool.handler is None:
            raise UnknownToolError(name)
        return tool.handler(**arguments)

    def is_finish(self, name: str) -> bool:
        return name == FINISH_TOOL

    # ─── 핸들러 — 모듈 위임 ───

    def _search_memory(self, query: str, top_k: int = 5) -> str:
        return self._memory.to_prompt(query=query, top_k=top_k)

    def _search_knowledge(self, query: str) -> str:
        result = self._knowledge.search_relevant(query=query)
        return result or f"'{query}'에 대한 설정 없음"

    def _search_fewshot(self, query: str, use_emotion: bool = True) -> str:
        emotions = self._emotion.get_state() if use_emotion else None
        result = self._fewshot.to_prompt(
            query=query, emotions=emotions, token_budget=TOOL_TOKEN_BUDGET
        )
        return result or f"'{query}'와 비슷한 예시 없음"

    def _get_history(self, n: int = 10) -> str:
        return self._history.to_prompt(n=n) or "최근 대화 없음"

    # ─── 기본 상태 — 도구 없이 항상 아는 것 (REQ-RA-70) ───

    def baseline(self) -> dict[str, str]:
        """뇌와 Stage 2가 함께 보는 상태. 어긋나면 톤과 판단이 따로 논다."""
        return {
            "emotion": self._emotion.to_prompt(),
            "history": self._history.to_prompt(n=BASELINE_HISTORY_TURNS),
            # 배경지식은 캐릭터의 상식이다. 판단할 때도, 말할 때도 알고 있어야 한다.
            "knowledge": self._knowledge.base_text(),
        }

    def knowledge_index(self) -> str:
        """뇌 전용. 무엇을 **더 찾아볼 수 있는지**의 목록이지 발화 재료가 아니다."""
        return self._knowledge.to_index()
