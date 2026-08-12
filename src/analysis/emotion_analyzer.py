"""대화를 읽고 감정 상태의 변화를 제안한다.

`EmotionModule`이 하던 LLM 상호작용을 옮겨온 자리다. 모듈은 이제 `EmotionAnalysis`만
받으므로, 블렌딩·제거 규칙을 LLM 없이 테스트할 수 있다 (REQ-14-4).

**제안일 뿐 적용이 아니다.** 무엇을 받아들일지(범위 검증·블렌딩 비율)는
도메인 모듈이 정한다.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmotionAnalysis:
    """감정 변화 제안.

    `significant`가 False면 나머지 필드는 무시된다 — 일상적 대화에서 감정이
    출렁이지 않게 하려는 장치다.
    """

    significant: bool = False
    emotions: dict[str, float] = field(default_factory=dict)
    remove: list[str] = field(default_factory=list)


# 응답 형식은 호출 흐름과 분리한다.
_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "emotion_update",
        "schema": {
            "type": "object",
            "properties": {
                "emotions": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "number",
                        "maximum": 1.0,
                        "minimum": 0.0,
                    },
                },
                "remove": {"type": "array", "items": {"type": "string"}},
                "significant": {"type": "boolean"},
            },
            "required": ["significant"],
        },
    },
}


class EmotionAnalyzer:
    """대화와 현재 감정 상태를 주고 변화를 제안받는다."""

    def __init__(self, client, on_prompt: Callable[[str, str], None] | None = None):
        self._client = client
        self._on_prompt = on_prompt

    def analyze(
        self,
        user_input: str,
        character_response: str,
        current_emotions: dict[str, float],
        history_context: str = "",
    ) -> EmotionAnalysis:
        """변화 제안을 반환한다. 거부·파싱 실패는 '변화 없음'으로 처리한다."""
        client = self._client
        current_emotions_str = (
            json.dumps(current_emotions, ensure_ascii=False) if current_emotions else "{}"
        )

        prompt = f"""캐릭터의 감정 상태를 업데이트하세요. 변화가 미미하면 현재 상태를 유지합니다.

{history_context}

사용자: {user_input}
캐릭터: {character_response}

현재 감정 상태:
{current_emotions_str}

다음 JSON 형식으로 반환하세요:
{{
    "emotions": {{
        "감정이름": 0.0~1.0,
        ...
    }},
    "remove": ["제거할 감정 이름", ...],
    "significant": true/false
}}

규칙:
- 감정이 없는 상태(빈 {{}})가 기본값입니다. 중립=정상 상태입니다
- significant가 true일 때만 emotions/remove를 채우세요
- 일상적 대화(안부, 짧은 대답, 정보 교환)는 significant=false로 반환하세요
- 감정 변화가 명확할 때만 significant=true: 감동, 분노, 슬픔, 큰 기쁨, 충격 등
- 감정 이름은 자유롭게 정하세요 (예: 행복, 슬픔, 분노, 설렘, 피로, 향수 등)
- 값은 0.0에서 1.0 사이
- 현재 감정 중 이 대화로 인해 완전히 사라진 것만 remove에 포함하세요
- 애매하면 유지하세요. 감정은 쉽게 변하지 않습니다
- 대부분의 대화에서 significant=false여야 합니다"""

        if self._on_prompt:
            self._on_prompt("emotion", prompt)

        result = client.call_llm(
            messages=[
                {"role": "system", "content": "감정 분석기. JSON만 반환하세요."},
                {"role": "user", "content": prompt},
            ],
            tools=[],
            use_stream=False,
            mute=True,
            response_format=_ANALYSIS_SCHEMA,
        ).content

        return _parse(result)


def _parse(result: str) -> EmotionAnalysis:
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, AttributeError, TypeError):
        # 거부든 형식 오류든 이 층에서는 "제안 없음"으로 같다.
        # 거부 사실 자체는 계측(`CallRecord.refused`)이 이미 남긴다.
        return EmotionAnalysis()

    if not data.get("significant", False):
        return EmotionAnalysis(significant=False)

    return EmotionAnalysis(
        significant=True,
        emotions=data.get("emotions", {}) or {},
        remove=data.get("remove", []) or [],
    )
