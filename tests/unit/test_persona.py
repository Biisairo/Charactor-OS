"""PersonaModule 단위 테스트."""

from __future__ import annotations

import pytest
import yaml

from src.modules.persona import PersonaModule

# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


class TestLoad:
    """YAML 파싱 및 검증."""

    def test_load_parses_yaml_into_dict(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        data = mod.load()

        assert isinstance(data, dict)
        assert data["name"] == "홍길동"
        assert "identity" in data
        assert "personality" in data
        assert "speaking_style" in data

    def test_load_returns_same_data_as_file(self, persona_path: str) -> None:
        """YAML 파일 내용과 load() 결과가 일치한다."""
        with open(persona_path, encoding="utf-8") as f:
            expected = yaml.safe_load(f)

        mod = PersonaModule(persona_path)
        data = mod.load()

        assert data == expected

    def test_load_raises_file_not_found(self, tmp_path) -> None:
        mod = PersonaModule(str(tmp_path / "nonexistent.yaml"))
        with pytest.raises(FileNotFoundError):
            mod.load()

    def test_load_raises_value_error_when_name_missing(self, tmp_path) -> None:
        """name 필드가 없으면 ValueError."""
        bad_file = tmp_path / "no_name.yaml"
        bad_file.write_text(yaml.dump({"identity": "테스트"}), encoding="utf-8")

        mod = PersonaModule(str(bad_file))
        with pytest.raises(ValueError, match="name"):
            mod.load()

    def test_load_raises_value_error_when_empty(self, tmp_path) -> None:
        """빈 YAML 파일이면 ValueError."""
        bad_file = tmp_path / "empty.yaml"
        bad_file.write_text("", encoding="utf-8")

        mod = PersonaModule(str(bad_file))
        with pytest.raises(ValueError):
            mod.load()


# ---------------------------------------------------------------------------
# to_system_prompt()
# ---------------------------------------------------------------------------


class TestToSystemPrompt:
    """시스템 프롬프트 변환."""

    def test_contains_name(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        prompt = mod.to_system_prompt()

        assert "홍길동" in prompt

    def test_contains_identity(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        prompt = mod.to_system_prompt()

        assert "아버지를 아버지라 부르지 못하는" in prompt

    def test_contains_age_and_occupation(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        prompt = mod.to_system_prompt()

        assert "나이: 25" in prompt
        assert "직업: 의적" in prompt

    def test_auto_loads_if_not_loaded(self, persona_path: str) -> None:
        """load()를 호출하지 않아도 to_system_prompt()가 내부적으로 로드한다."""
        mod = PersonaModule(persona_path)
        prompt = mod.to_system_prompt()

        assert "홍길동" in prompt

    def test_contains_personality_traits(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        prompt = mod.to_system_prompt()

        assert "용감한" in prompt
        assert "정의로운" in prompt

    def test_contains_speaking_style(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        prompt = mod.to_system_prompt()

        assert "존댓말과 반말을 섞어" in prompt

    def test_contains_all_speaking_style_fields(self, persona_path: str) -> None:
        """tone·vocabulary·sentence_pattern·emojis도 프롬프트에 실린다.

        summary·fillers·endings만 나가던 시절에는 말투 서술의 절반이
        로드만 되고 버려졌다. 캐릭터마다 톤·어휘가 다른 것이 핵심인데,
        그 차이가 프롬프트에 도달하지 않았다.
        """
        mod = PersonaModule(persona_path)
        mod.load()
        prompt = mod.to_system_prompt()

        assert "차분하지만 감정이 올라올 때는 날카로워짐" in prompt  # tone
        assert "평소에는 일상어, 화가 나면 고어체가 섞임" in prompt  # vocabulary
        assert "짧은 문장 위주, 핵심만 말함" in prompt  # sentence_pattern
        assert "거의 사용 안함" in prompt  # emojis

    def test_omits_absent_speaking_style_fields(self, tmp_path) -> None:
        """없는 필드는 라벨도 나가지 않는다 (빈 항목 방지)."""
        path = tmp_path / "minimal.yaml"
        path.write_text(
            yaml.dump(
                {"name": "테스트", "speaking_style": {"summary": "간결함"}},
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        prompt = PersonaModule(str(path)).to_system_prompt()

        assert "간결함" in prompt
        for label in ["- 톤:", "- 어휘:", "- 문장:", "- 이모지:"]:
            assert label not in prompt

    def test_contains_values(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        prompt = mod.to_system_prompt()

        assert "정의" in prompt
        assert "가족" in prompt

    def test_contains_backstory(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        prompt = mod.to_system_prompt()

        assert "조선시대 양반 가문의 서자" in prompt


# ---------------------------------------------------------------------------
# get_emotion_triggers()
# ---------------------------------------------------------------------------


class TestGetEmotionTriggers:
    """감정 트리거 목록."""

    def test_returns_list(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        triggers = mod.get_emotion_triggers()

        assert isinstance(triggers, list)

    def test_returns_non_empty_list(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        triggers = mod.get_emotion_triggers()

        assert len(triggers) > 0

    def test_each_trigger_has_expected_fields(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        triggers = mod.get_emotion_triggers()

        for trigger in triggers:
            assert "keyword" in trigger
            assert "emotion" in trigger
            assert "intensity" in trigger

    def test_triggers_match_yaml_data(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        data = mod.load()
        triggers = mod.get_emotion_triggers()

        assert triggers == data["emotion_triggers"]

    def test_returns_empty_list_when_key_missing(self, tmp_path) -> None:
        """emotion_triggers 키가 없으면 빈 리스트."""
        minimal = tmp_path / "minimal.yaml"
        minimal.write_text(yaml.dump({"name": "테스트"}), encoding="utf-8")

        mod = PersonaModule(str(minimal))
        mod.load()
        triggers = mod.get_emotion_triggers()

        assert triggers == []


# ---------------------------------------------------------------------------
# get_examples()
# ---------------------------------------------------------------------------


class TestGetExamples:
    """내장 few-shot 예시."""

    def test_returns_list(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        examples = mod.get_examples()

        assert isinstance(examples, list)

    def test_returns_non_empty_list(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        examples = mod.get_examples()

        assert len(examples) > 0

    def test_each_example_has_user_and_character(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        mod.load()
        examples = mod.get_examples()

        for ex in examples:
            assert "user" in ex
            assert "character" in ex

    def test_examples_match_yaml_data(self, persona_path: str) -> None:
        mod = PersonaModule(persona_path)
        data = mod.load()
        examples = mod.get_examples()

        assert examples == data["examples"]

    def test_returns_empty_list_when_key_missing(self, tmp_path) -> None:
        """examples 키가 없으면 빈 리스트."""
        minimal = tmp_path / "minimal.yaml"
        minimal.write_text(yaml.dump({"name": "테스트"}), encoding="utf-8")

        mod = PersonaModule(str(minimal))
        mod.load()
        examples = mod.get_examples()

        assert examples == []
