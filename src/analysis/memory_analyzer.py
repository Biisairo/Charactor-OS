"""대화에서 기억할 사실을 뽑고, 기존 기억과의 관계를 판정한다.

`MemoryModule`이 하던 LLM 상호작용을 옮겨온 자리다. 모듈은 이제 이 층이 돌려주는
`MemoryCandidate`와 `Classification`만 받으므로, 도메인 규칙(병합·가중치·중복 처리)을
LLM 없이 테스트할 수 있다 (REQ-14-4).

프롬프트 **내용은 옮기기만 했다.** 튜닝은 TASK-08·09 소관이다.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from src.prompts.untrusted import MEMORY, quote

Classification = Literal["IDENTICAL", "SIMILAR", "DIFFERENT"]

DEFAULT_IMPORTANCE = 0.5


@dataclass(frozen=True)
class MemoryCandidate:
    """대화에서 뽑아낸, 저장 후보가 되는 사실 하나."""

    content: str
    importance: float = DEFAULT_IMPORTANCE


# 응답 형식은 호출 흐름과 분리한다 — 스키마가 본문에 섞이면 흐름이 묻힌다.
_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "importance": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
            "required": ["memories"],
        },
    },
}

_CLASSIFICATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_conflict",
        "schema": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "enum": ["SAME", "CONTRADICT", "DIFFERENT"],
                },
            },
            "required": ["classification"],
        },
    },
}


class MemoryExtractor:
    """대화에서 사용자에 대한 구체적 사실을 뽑는다."""

    def __init__(self, client, on_prompt: Callable[[str, str], None] | None = None):
        self._client = client
        # 프롬프트를 디버그 패널로 흘려보내는 훅. 프롬프트를 소유한 이 층이
        # 갖는 것이 맞다 — 도메인 모듈 API를 관통하던 배관을 여기로 걷어왔다.
        self._on_prompt = on_prompt

    def extract(
        self, user_input: str, character_response: str, history_context: str = ""
    ) -> list[MemoryCandidate]:
        """저장 후보를 반환한다. 거부·파싱 실패는 빈 목록이 된다."""
        client = self._client

        prompt = f"""다음 대화에서 **사용자에 대한 구체적인 사실**만 추출하세요.

{history_context}

이번 대화입니다.
{quote(user_input, attrs={"화자": "사용자"})}
{quote(character_response, attrs={"화자": "캐릭터"})}

다음 JSON 형식으로 반환하세요:
{{
    "memories": [
        {{
            "content": "기억할 내용",
            "importance": 0.0~1.0
        }}
    ]
}}

기억할 수 있는 것 (구체적 사실):
- 이름, 나이, 직업, 거주지
- 좋아하는/싫어하는 것
- 가족, 반려동물
- 특별한 경험, 사건
- 고민, 목표

기억하면 안 되는 것:
- 대화 스타일, 패턴
- 감정 상태 (별도 관리됨)
- 일반적인 관심표현
- 모호한 추론

규칙:
- 이미 알려진 정보와 중복되면 저장하지 않음
- 구체적인 사실만 저장 (추론 X)
- 추출할 정보가 없으면 빈 배열 반환"""

        if self._on_prompt:
            self._on_prompt("memory", prompt)

        result = client.call_llm(
            messages=[
                {"role": "system", "content": "기억 추출기. JSON만 반환하세요."},
                {"role": "user", "content": prompt},
            ],
            tools=[],
            use_stream=False,
            mute=True,
            response_format=_EXTRACTION_SCHEMA,
        ).content

        return _parse_candidates(result)


class ConflictClassifier:
    """새 기억이 기존 기억과 같은 정보인지 판정한다."""

    def __init__(self, client, on_prompt: Callable[[str, str], None] | None = None):
        self._client = client
        self._on_prompt = on_prompt

    def classify(self, existing_content: str, content: str) -> Classification:
        """판정할 수 없으면 `DIFFERENT` — 기존 기억을 덮어쓰지 않는 쪽이 안전하다."""
        client = self._client

        prompt = f"""다음 두 기억을 비교하세요.

기존 기억:
{quote(existing_content, kind=MEMORY)}
새 기억:
{quote(content, kind=MEMORY)}

다음 중 하나로 분류하세요:
- IDENTICAL: 같은 정보
- SIMILAR: 관련 있지만 다른 정보
- DIFFERENT: 완전히 다른 정보

JSON으로 반환: {{"classification": "IDENTICAL|SIMILAR|DIFFERENT"}}"""

        if self._on_prompt:
            self._on_prompt("memory_conflict", prompt)

        result = client.call_llm(
            messages=[
                {"role": "system", "content": "기억 분류기. JSON만 반환하세요."},
                {"role": "user", "content": prompt},
            ],
            tools=[],
            use_stream=False,
            mute=True,
            response_format=_CLASSIFICATION_SCHEMA,
        ).content

        return _parse_classification(result)


def _parse_candidates(result: str) -> list[MemoryCandidate]:
    try:
        raw = json.loads(result).get("memories", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []

    candidates = []
    for item in raw:
        content = item.get("content", "")
        if content:
            candidates.append(
                MemoryCandidate(
                    content=content, importance=item.get("importance", DEFAULT_IMPORTANCE)
                )
            )
    return candidates


def _parse_classification(result: str) -> Classification:
    try:
        value = json.loads(result).get("classification", "DIFFERENT")
    except (json.JSONDecodeError, AttributeError, TypeError):
        return "DIFFERENT"

    return value if value in ("IDENTICAL", "SIMILAR", "DIFFERENT") else "DIFFERENT"
