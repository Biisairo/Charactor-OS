"""POST /api/characters — 질문지(static_data)로 생성 시 static/ 전체를 채운다.

`create_character`는 `characters/` 전역 경로에 쓴다. 테스트가 실제 저장소를
건드리지 않도록 라우터 모듈의 `CHARACTERS_DIR`을 임시 디렉토리로 치환한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.api import schemas
from src.api.routers import characters as chars_router


def _draft() -> dict:
    return {
        "persona": {
            "name": "테스트캐릭",
            "identity": "질문지로 만든 캐릭터",
            "age": "20대 초반",
            "gender": "여성",
            "occupation": "모험가",
            "personality": {
                "traits": ["밝은"],
                "big5": {
                    "openness": 0.8,
                    "conscientiousness": 0.5,
                    "extraversion": 0.7,
                    "agreeableness": 0.6,
                    "neuroticism": 0.3,
                },
            },
            "speaking_style": {
                "summary": "밝은 말투",
                "tone": "경쾌한",
                "vocabulary": "일상어",
                "sentence_pattern": "짧은 문장",
                "fillers": ["어"],
                "emojis": "적게 사용",
                "endings": ["~야"],
            },
            "values": ["자유"],
            "backstory": "작은 마을에서 자랐다.",
            "likes": ["산책"],
            "dislikes": ["거짓말"],
            "fears": ["어둠"],
            "goals": ["세계 일주"],
            "behavior": {
                "situations": [{"trigger": "고민", "action": "경청"}],
                "topics": [{"name": "여행", "stance": "신나함"}],
                "rules": ["거짓말하지 않는다"],
            },
            "emotion_triggers": [{"keyword": "어둠", "emotion": "불안", "intensity": 0.6}],
            "relationships": [{"target": "사용자", "type": "동료", "description": "함께 모험"}],
            "inner_world": {
                "current_thought": "다음 행선지",
                "hidden_feelings": "두려움",
                "wants_to_say": "고맙다",
            },
            "examples": [{"user": "안녕", "character": "안녕!", "scenario": "인사"}],
        },
        "knowledge": {
            "world": {
                "name": "모험의 땅",
                "era": "중세",
                "description": "마법이 존재하는 세계",
                "rules": ["마법은 희귀하다"],
                "technology_level": "전근대",
                "social_structure": "왕국-영지",
            },
            "locations": [
                {
                    "name": "고향 마을",
                    "description": "작은 마을",
                    "significance": "출생지",
                    "characters_present": ["테스트캐릭"],
                }
            ],
            "relationships": [
                {
                    "from": "테스트캐릭",
                    "to": "사용자",
                    "type": "동료",
                    "sentiment": "신뢰",
                    "description": "모험 동료",
                    "strength": 0.8,
                }
            ],
            "timeline": [
                {
                    "time": "10세",
                    "event": "마을을 떠남",
                    "characters_involved": ["테스트캐릭"],
                    "impact": "모험의 시작",
                }
            ],
            "freeform": "여행 일지: 첫날 무사히 출발.",
        },
        "examples": {
            "greeting": {"tag": "인사", "examples": [{"user": "안녕", "character": "반가워!"}]},
            "comfort": {
                "tag": "위로",
                "examples": [{"user": "힘들어", "character": "괜찮아, 내가 있잖아.", "emotion_state": ["슬픔"]}],
            },
        },
    }


@pytest.fixture
def isolated_characters_dir(monkeypatch, tmp_path: Path) -> Path:
    """characters/ 를 임시 디렉토리로 치환한다 — 실제 저장소를 건드리지 않도록."""
    base = tmp_path / "characters"
    base.mkdir()
    monkeypatch.setattr(chars_router, "CHARACTERS_DIR", base)
    return base


class TestCreateWithStaticData:
    def test_writes_all_filled_sections(self, isolated_characters_dir: Path):
        draft = _draft()
        req = schemas.CreateCharacterRequest(
            name="테스트캐릭",
            identity=draft["persona"]["identity"],
            static_data=schemas.CharacterStaticData(**draft),
        )

        resp = chars_router.create_character(req)

        assert resp == {"status": "ok", "character": "테스트캐릭"}
        static = isolated_characters_dir / "테스트캐릭" / "static"

        # persona — 질문지 응답이 그대로 반영된다
        persona = yaml.safe_load((static / "persona.yaml").read_text(encoding="utf-8"))
        assert persona["name"] == "테스트캐릭"
        assert persona["personality"]["traits"] == ["밝은"]
        assert persona["inner_world"]["hidden_feelings"] == "두려움"

        # knowledge — type 필드가 붙어 KnowledgeModule 스키마로 저장된다
        world = yaml.safe_load((static / "knowledge" / "world.yaml").read_text(encoding="utf-8"))
        assert world["type"] == "world"
        assert world["name"] == "모험의 땅"
        locations = yaml.safe_load((static / "knowledge" / "locations.yaml").read_text(encoding="utf-8"))
        assert locations["type"] == "locations"
        assert locations["locations"][0]["name"] == "고향 마을"
        rel = yaml.safe_load((static / "knowledge" / "relationships.yaml").read_text(encoding="utf-8"))
        assert rel["type"] == "relationships"
        assert rel["relationships"][0]["strength"] == 0.8
        timeline = yaml.safe_load((static / "knowledge" / "timeline.yaml").read_text(encoding="utf-8"))
        assert timeline["type"] == "timeline"
        assert timeline["events"][0]["impact"] == "모험의 시작"
        assert (static / "knowledge" / "notes.md").read_text(encoding="utf-8") == (
            "여행 일지: 첫날 무사히 출발."
        )

        # examples — 채운 시나리오만 파일로 남는다
        greeting = yaml.safe_load((static / "examples" / "greeting.yaml").read_text(encoding="utf-8"))
        assert greeting["tag"] == "인사"
        comfort = yaml.safe_load((static / "examples" / "comfort.yaml").read_text(encoding="utf-8"))
        assert comfort["examples"][0]["emotion_state"] == ["슬픔"]
        assert not (static / "examples" / "conflict.yaml").exists()

    def test_skipped_sections_create_no_files(self, isolated_characters_dir: Path):
        """빈 섹션은 파일을 남기지 않는다 — 빈 지식이 로드되지 않게."""
        draft = _draft()
        draft["knowledge"] = {
            "world": {"name": "", "era": "", "description": "", "rules": [], "technology_level": "", "social_structure": ""},
            "freeform": "   ",
        }
        draft["examples"] = {}
        req = schemas.CreateCharacterRequest(
            name="테스트캐릭",
            static_data=schemas.CharacterStaticData(**draft),
        )

        chars_router.create_character(req)

        static = isolated_characters_dir / "테스트캐릭" / "static"
        assert not (static / "knowledge" / "world.yaml").exists()
        assert not (static / "knowledge" / "notes.md").exists()
        assert not (static / "examples" / "greeting.yaml").exists()


class TestCreateWithoutStaticData:
    def test_writes_template(self, isolated_characters_dir: Path):
        """기존 동작 — 이름·소개만으로 생성하면 최소 템플릿을 쓴다."""
        req = schemas.CreateCharacterRequest(name="빈캐릭", identity="테스트")

        chars_router.create_character(req)

        static = isolated_characters_dir / "빈캐릭" / "static"
        persona = yaml.safe_load((static / "persona.yaml").read_text(encoding="utf-8"))
        assert persona["name"] == "빈캐릭"
        assert persona["identity"] == "테스트"
        assert not (static / "knowledge" / "world.yaml").exists()
        assert not (static / "examples" / "greeting.yaml").exists()


class TestDraftRoundTrip:
    def test_draft_returns_filled_sections(self, isolated_characters_dir: Path):
        """생성한 캐릭터를 질문지 형태로 다시 읽을 수 있어야 한다 — 재오픈용."""
        draft = _draft()
        chars_router.create_character(
            schemas.CreateCharacterRequest(
                name="테스트캐릭",
                static_data=schemas.CharacterStaticData(**draft),
            )
        )

        out = chars_router.get_character_draft("테스트캐릭")

        assert out["persona"]["name"] == "테스트캐릭"
        assert out["persona"]["inner_world"]["hidden_feelings"] == "두려움"
        # type: 필드는 프론트 응답에서 빠져야 한다 — CharacterDraft.world는 type이 없다
        assert "type" not in out["knowledge"]["world"]
        assert out["knowledge"]["world"]["name"] == "모험의 땅"
        assert out["knowledge"]["locations"][0]["name"] == "고향 마을"
        assert out["knowledge"]["freeform"] == "여행 일지: 첫날 무사히 출발."
        assert out["examples"]["greeting"]["tag"] == "인사"

    def test_draft_of_unknown_character_404(self, isolated_characters_dir: Path):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            chars_router.get_character_draft("없는캐릭")
        assert exc.value.status_code == 404


class TestUpdateStatic:
    def test_overwrites_and_removes_emptied_sections(self, isolated_characters_dir: Path):
        """수정 모드 — 비운 섹션의 이전 파일이 지워져야 한다."""
        draft = _draft()
        chars_router.create_character(
            schemas.CreateCharacterRequest(
                name="테스트캐릭",
                static_data=schemas.CharacterStaticData(**draft),
            )
        )

        # 세계관·인사 예시만 남기고 나머지는 비운다
        slim = _draft()
        slim["knowledge"] = {
            "world": slim["knowledge"]["world"],
        }
        slim["examples"] = {
            "greeting": slim["examples"]["greeting"],
        }
        resp = chars_router.update_character_static(
            "테스트캐릭",
            schemas.CharacterStaticData(**slim),
        )
        assert resp == {"status": "ok", "character": "테스트캐릭"}

        static = isolated_characters_dir / "테스트캐릭" / "static"
        assert (static / "knowledge" / "world.yaml").exists()
        assert not (static / "knowledge" / "locations.yaml").exists()
        assert not (static / "knowledge" / "relationships.yaml").exists()
        assert not (static / "knowledge" / "timeline.yaml").exists()
        assert not (static / "knowledge" / "notes.md").exists()
        assert (static / "examples" / "greeting.yaml").exists()
        assert not (static / "examples" / "comfort.yaml").exists()

        # persona는 항상 응답 전체로 덮어쓴다
        persona = yaml.safe_load((static / "persona.yaml").read_text(encoding="utf-8"))
        assert persona["name"] == "테스트캐릭"

    def test_update_unknown_character_404(self, isolated_characters_dir: Path):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            chars_router.update_character_static(
                "없는캐릭", schemas.CharacterStaticData()
            )
        assert exc.value.status_code == 404
