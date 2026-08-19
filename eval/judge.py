"""LLM-as-judge — 캐릭터 응답을 3개 축으로 채점한다 (REQ-01-3).

판정 프롬프트 조립은 순수 함수로 분리하여 LLM 없이 검증할 수 있게 했다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eval.dataset import GoldenCase
from eval.scoring import AXES, CaseScore, JudgeParseError, parse_judge_response
from src.character_layout import CharacterLayout
from src.modules.knowledge import KnowledgeModule
from src.modules.persona import PersonaModule
from src.prompts.untrusted import quote

# 판정자가 축을 빠뜨리거나 형식을 어기는 일이 있다. 사례를 버리면 표본이 줄고
# 편향이 생기므로, 형식을 다시 일러주며 재시도한다.
MAX_JUDGE_ATTEMPTS = 3

RETRY_NUDGE = (
    "\n\n[재요청] 직전 응답의 형식이 올바르지 않았습니다. "
    f"반드시 {', '.join(AXES)} 세 축을 모두 포함한 JSON 객체만 출력하세요."
)


@dataclass(frozen=True)
class CharacterProfile:
    """판정 기준을 세우는 데 필요한 캐릭터 정보 (REQ-05-5).

    판정 프롬프트에 '조선 시대 홍길동'이 박혀 있으면, 다른 캐릭터는 자기
    설정을 지켜도 '조선 말투가 아니다'라는 이유로 감점된다. 평가 기준을
    캐릭터 자산에서 파생시켜야 평가가 캐릭터에 종속되지 않는다.
    """

    name: str = ""
    era: str = ""
    speech: str = ""

    @property
    def described(self) -> str:
        """판정 프롬프트에 넣을 캐릭터 서술. 정보가 없으면 빈 문자열."""
        if not self.name:
            return ""
        parts = [f"평가 대상은 '{self.name}'을(를) 연기하는 대화 시스템입니다."]
        if self.era:
            parts.append(f"이 캐릭터가 속한 시대·배경은 다음과 같습니다: {self.era}")
        if self.speech:
            parts.append(f"이 캐릭터의 말투는 다음과 같이 정의되어 있습니다: {self.speech}")
        return "\n".join(parts)


def load_character_profile(character_dir: Path | str) -> CharacterProfile:
    """캐릭터 자산에서 판정 기준을 읽는다. 자산이 없거나 비어도 실패하지 않는다."""
    layout = CharacterLayout.of(character_dir)

    persona = PersonaModule(str(layout.persona_path))
    data = persona.load()
    speaking = data.get("speaking_style") or {}

    knowledge = KnowledgeModule(str(layout.knowledge_dir))
    knowledge.load_all()

    return CharacterProfile(
        name=data.get("name", ""),
        era=knowledge.era(),
        speech=speaking.get("summary", ""),
    )


def build_judge_system_prompt(profile: CharacterProfile | None = None) -> str:
    """판정 시스템 프롬프트를 만든다 (순수 함수).

    profile이 없으면 캐릭터 중립 기준으로 채점한다 — 설정 자체를 기준으로
    삼으므로 특정 시대를 가정하지 않는다.
    """
    described = profile.described if profile else ""
    intro = f"\n{described}\n" if described else ""

    return f"""당신은 캐릭터 챗봇의 응답 품질을 평가하는 심사관입니다.
{intro}
아래 세 축을 각각 1~5점으로 채점하세요.
채점 기준은 **그 캐릭터의 설정**이며, 특정 시대나 말투를 미리 가정하지 마세요.

1. tone (말투 일관성)
   5 = 캐릭터에 정의된 말투가 일관되게 유지됨
   3 = 대체로 유지되나 캐릭터에 맞지 않는 어투가 일부 섞임
   1 = 캐릭터 말투가 무너지고 일반 어시스턴트처럼 응답함

2. worldview (세계관 정합성)
   5 = 캐릭터의 시대·배경·설정에 어긋남이 없음
   3 = 큰 모순은 없으나 모호하거나 설정에 없는 내용이 섞임
   1 = 캐릭터의 시대·배경에 맞지 않거나 설정과 명백히 모순됨

3. memory (기억 활용 적절성)
   5 = 앞선 대화에서 얻은 정보를 정확히 활용함.
       참조할 기억이 없는 상황이라면, 억지로 기억을 지어내지 않은 것도 5점
   3 = 부분적으로 활용하거나 모호하게 언급함
   1 = 알고 있어야 할 정보를 틀리게 답하거나, 없는 기억을 지어냄

기대 동작이 함께 주어집니다. 기대와 다르면 낮게 채점하세요.
관대하게 주지 마세요. 결함이 있으면 반드시 점수에 반영하세요.

반드시 아래 JSON 형식으로만 답하세요. 다른 설명을 덧붙이지 마세요.

{{
  "tone": {{"score": <1-5>, "reason": "<한 문장>"}},
  "worldview": {{"score": <1-5>, "reason": "<한 문장>"}},
  "memory": {{"score": <1-5>, "reason": "<한 문장>"}}
}}"""


def build_judge_prompt(case: GoldenCase, response: str) -> str:
    """판정자에게 보낼 사용자 메시지를 만든다 (순수 함수)."""
    parts = [f"[범주] {case.category}"]

    if case.setup:
        prior = "\n".join(f"  - {utterance}" for utterance in case.setup)
        parts.append(f"[앞선 대화에서 사용자가 밝힌 내용]\n{prior}")

    # 사례 입력·기대 동작은 데이터셋 저작물이라 신뢰한다. 캐릭터 응답만
    # 모델이 만든 것이므로 경계 안에 둔다 — 응답에 `[기대 동작]`을 심어
    # 점수를 흔드는 경로를 막는다 (SPEC-10 REQ-10-18).
    quoted = quote(response, attrs={"화자": "캐릭터"})
    parts.append(f"[사용자 입력]\n{case.input}")
    parts.append(f"[캐릭터 응답]\n{quoted}")
    parts.append(f"[기대 동작]\n{case.expectation}")

    return "\n\n".join(parts)


@dataclass
class Judge:
    """판정자. client는 call_llm(messages, tools, use_stream, mute, ...)를 제공해야 한다."""

    client: object
    profile: CharacterProfile | None = None

    def score(self, case: GoldenCase, response: str) -> CaseScore:
        """사례를 채점한다. 형식 오류는 재시도하고, 끝내 실패하면 예외를 올린다."""
        prompt = build_judge_prompt(case, response)
        system_prompt = build_judge_system_prompt(self.profile)
        last_error: JudgeParseError | None = None

        for attempt in range(MAX_JUDGE_ATTEMPTS):
            result = self.client.call_llm(  # type: ignore[attr-defined]
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt if attempt == 0 else prompt + RETRY_NUDGE},
                ],
                tools=[],
                use_stream=False,
                mute=True,
                response_format={"type": "json_object"},
            )
            try:
                scores, reasons = parse_judge_response(result.content)
            except JudgeParseError as exc:
                last_error = exc
                continue

            return CaseScore(
                case_id=case.id,
                category=case.category,
                scores=scores,
                reasons=reasons,
            )

        raise JudgeParseError(f"{MAX_JUDGE_ATTEMPTS}회 시도 후 실패 — {last_error}")


class StubJudge:
    """비용 없이 파이프라인을 점검하기 위한 판정자 (--dry-run).

    응답 길이에 따라 결정론적으로 점수를 만든다. 품질 평가로서 의미는 없고,
    실행 경로·집계·저장이 동작하는지 확인하는 용도다.
    """

    def score(self, case: GoldenCase, response: str) -> CaseScore:
        base = 3 + (len(response) % 3)
        scores = {axis: max(1, min(5, base - i)) for i, axis in enumerate(AXES)}
        return CaseScore(
            case_id=case.id,
            category=case.category,
            scores=scores,
            reasons=dict.fromkeys(AXES, "dry-run 더미 판정"),
        )
