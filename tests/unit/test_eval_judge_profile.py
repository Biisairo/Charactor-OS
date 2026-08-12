"""판정 기준이 캐릭터 자산에서 파생되는지 검증한다 (REQ-05-5).

판정 프롬프트에 '조선 시대 홍길동'이 상수로 박혀 있던 동안에는, 다른 캐릭터가
자기 설정을 완벽히 지켜도 '조선 말투가 아니다'라는 이유로 감점될 수 있었다.
평가 하네스가 캐릭터 1종에 종속되어 있었다는 뜻이다.
"""

from __future__ import annotations

import pytest

from eval.dataset import GoldenCase
from eval.judge import (
    CharacterProfile,
    Judge,
    build_judge_system_prompt,
    load_character_profile,
)
from src.llm.client import TrimmedMessage

CASE = GoldenCase(id="c1", category="greeting", input="안녕", expectation="인사한다")
VALID = '{"tone": 4, "worldview": 4, "memory": 4}'


class _CapturingClient:
    """판정자에게 전달된 시스템 프롬프트를 잡아둔다."""

    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    def call_llm(self, messages, tools=None, use_stream=False, mute=True, **kwargs):
        self.system_prompts.append(messages[0]["content"])
        return TrimmedMessage(
            content=VALID, role="assistant", reasoning_content="", tool_calls=[], usage=None
        )


class TestBuildJudgeSystemPrompt:
    def test_omits_character_specifics_without_profile(self):
        """profile이 없으면 특정 시대·인물을 가정하지 않는다."""
        prompt = build_judge_system_prompt()

        assert "조선" not in prompt
        assert "홍길동" not in prompt

    def test_includes_profile_fields(self):
        profile = CharacterProfile(
            name="소민찌",
            era="2020년대 후반 대한민국 서울",
            speech="빠른 구어체 반말",
        )

        prompt = build_judge_system_prompt(profile)

        assert "소민찌" in prompt
        assert "2020년대 후반 대한민국 서울" in prompt
        assert "빠른 구어체 반말" in prompt

    def test_no_character_is_hardcoded(self):
        """어떤 캐릭터의 프로필을 넣어도 다른 캐릭터가 새어들지 않는다."""
        prompt = build_judge_system_prompt(CharacterProfile(name="소민찌", era="현대"))

        assert "조선" not in prompt
        assert "홍길동" not in prompt

    def test_json_format_block_survives_formatting(self):
        """f-string 변환 후에도 판정 응답 형식 예시가 온전해야 한다."""
        prompt = build_judge_system_prompt()

        assert '"tone": {"score": <1-5>, "reason": "<한 문장>"}' in prompt


class TestLoadCharacterProfile:
    @pytest.mark.parametrize(
        ("character", "name", "era_fragment"),
        [
            ("hong-gil-dong", "홍길동", "조선"),
            ("han-so-min", "소민찌", "2020년대"),
        ],
    )
    def test_reads_from_character_assets(self, character, name, era_fragment):
        profile = load_character_profile(f"characters/{character}")

        assert profile.name == name
        assert era_fragment in profile.era
        assert profile.speech

    def test_missing_world_does_not_fail(self, tmp_path):
        """knowledge가 없어도 페르소나만으로 프로필이 만들어진다."""
        (tmp_path / "persona.yaml").write_text("name: 테스트\n", encoding="utf-8")

        profile = load_character_profile(tmp_path)

        assert profile.name == "테스트"
        assert profile.era == ""
        assert profile.described  # 이름만 있어도 서술은 만들어진다


class TestJudgeUsesProfile:
    def test_profile_reaches_the_judge_prompt(self):
        client = _CapturingClient()
        profile = CharacterProfile(name="소민찌", era="2020년대 후반", speech="구어체 반말")

        Judge(client=client, profile=profile).score(CASE, "왔어? 앉아 앉아")

        assert "소민찌" in client.system_prompts[0]
        assert "조선" not in client.system_prompts[0]

    def test_defaults_to_neutral_prompt(self):
        """profile 없이도 동작한다 (기존 호출부 호환)."""
        client = _CapturingClient()

        score = Judge(client=client).score(CASE, "반갑소")

        assert score.scores["tone"] == 4
        assert "홍길동" not in client.system_prompts[0]
