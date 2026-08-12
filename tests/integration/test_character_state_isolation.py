"""캐릭터별 상태 격리 (TASK-17 / REQ-17-1 ~ 17-3).

`memory_db_path`·`emotion_save_path`·`history_save_path`는 `character_dir`와
독립된 전역 설정이었다. `--character`는 캐릭터 디렉토리만 바꾸고 저장 경로는
그대로 두므로, **서로 다른 캐릭터가 같은 기억 DB를 공유했다.**

증상이 조용한 것이 문제였다 — 홍길동으로 만든 기억이 소민찌의 컨텍스트에
검색되어 들어가도 오류가 나지 않는다. 웹 UI의 캐릭터 전환 버튼이 실제로 이
경로를 밟는다.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.analysis import MemoryCandidate
from src.character_layout import CharacterLayout
from src.character_os import CharacterOS
from tests.conftest import MockClient, make_character_os

MEMORABLE = "내 이름은 박서준이오"


class _StubExtractor:
    """정해진 후보를 돌려준다. LLM 더블이 아니라 입력 고정 장치다."""

    def __init__(self, candidates: list[MemoryCandidate]):
        self._candidates = candidates

    def extract(self, user_input, character_response, history_context=""):
        return self._candidates


class _NeverConflicts:
    """충돌 판정을 타지 않도록 항상 '다른 기억'으로 분류한다."""

    def classify(self, existing_content, content):
        return "DIFFERENT"


def _make_two_characters(tmp_path: Path) -> tuple[Path, Path]:
    """실제 캐릭터 자산을 복사해 두 벌을 만든다."""
    source = Path("characters/hong-gil-dong")
    first = tmp_path / "characters" / "alpha"
    second = tmp_path / "characters" / "beta"
    shutil.copytree(source, first)
    shutil.copytree(source, second)
    return first, second


def _talk(character_dir: Path, text: str, reply: str) -> CharacterOS:
    """상태 경로를 지정하지 않고 대화한다 — 파생 경로가 쓰이는지 보기 위함이다."""
    client = MockClient()
    client.next_response.content = reply
    cos = make_character_os(
        character_dir,
        character_dir,  # make_character_os가 넘기는 명시 경로를 아래에서 지운다
        client,
        memory_db_path=None,
        emotion_save_path=None,
        history_save_path=None,
    )
    cos.chat(text)
    return cos


# ---------------------------------------------------------------------------
# REQ-17-1 — 캐릭터를 바꾸면 저장소도 함께 분리된다
# ---------------------------------------------------------------------------


class TestDerivedPaths:
    def test_state_paths_live_under_the_character(self, tmp_path: Path):
        first, second = _make_two_characters(tmp_path)

        cos_a = CharacterOS(character_dir=str(first), client=MockClient(), output=lambda _m: None)
        cos_b = CharacterOS(character_dir=str(second), client=MockClient(), output=lambda _m: None)

        assert cos_a.memory._db_path != cos_b.memory._db_path
        assert Path(cos_a.memory._db_path) == CharacterLayout.of(first).memory_db_path
        assert Path(cos_b.memory._db_path) == CharacterLayout.of(second).memory_db_path

    def test_explicit_paths_still_win(self, tmp_path: Path):
        """명시 경로는 그대로 존중된다 — 평가 하네스와 이전 설정의 호환 (REQ-17-2)."""
        first, _ = _make_two_characters(tmp_path)
        custom = tmp_path / "elsewhere" / "memories.db"

        cos = CharacterOS(
            character_dir=str(first),
            memory_db_path=str(custom),
            client=MockClient(),
            output=lambda _m: None,
        )

        assert Path(cos.memory._db_path) == custom


# ---------------------------------------------------------------------------
# REQ-17-3 — 두 캐릭터의 상태가 서로 섞이지 않는다
# ---------------------------------------------------------------------------


class TestStateDoesNotLeak:
    def test_memory_written_by_one_character_is_not_visible_to_the_other(self, tmp_path: Path):
        """alpha에 기억을 심고, beta를 새로 열어 그 기억이 보이는지 본다.

        `MockClient`는 기억을 추출하지 않으므로 대화만으로는 기억이 생기지
        않는다. 격리 자체를 재려면 기억을 결정론적으로 심어야 한다.
        """
        first, second = _make_two_characters(tmp_path)

        cos_a = CharacterOS(character_dir=str(first), client=MockClient(), output=lambda _m: None)
        cos_a.memory.update(
            user_input=MEMORABLE,
            character_response="반갑소, 박서준.",
            emotions={},
            extractor=_StubExtractor(
                [MemoryCandidate(content="사용자의 이름은 박서준", importance=0.9)]
            ),
            classifier=_NeverConflicts(),
        )
        cos_a.memory.save()
        assert cos_a.memory.snapshot_count() == 1, "전제: alpha에 기억이 심어져야 한다"

        cos_b = CharacterOS(character_dir=str(second), client=MockClient(), output=lambda _m: None)

        assert cos_b.memory.snapshot_count() == 0, (
            "다른 캐릭터가 만든 기억이 이 캐릭터의 저장소에서 로드되었다"
        )
        assert all("박서준" not in str(m) for m in cos_b.memory.search("박서준", top_k=50))

    def test_each_character_gets_its_own_state_directory(self, tmp_path: Path):
        first, second = _make_two_characters(tmp_path)

        _talk(first, MEMORABLE, "반갑소, 박서준.")
        _talk(second, "안녕하시오", "반갑소.")

        assert CharacterLayout.of(first).state_dir.exists()
        assert CharacterLayout.of(second).state_dir.exists()
        assert CharacterLayout.of(first).memory_db_path.exists()

    def test_history_does_not_leak(self, tmp_path: Path):
        first, second = _make_two_characters(tmp_path)

        _talk(first, MEMORABLE, "반갑소, 박서준.")
        cos_b = _talk(second, "안녕하시오", "반갑소.")

        turns = cos_b.history.get_recent(10)
        assert all(MEMORABLE not in str(t) for t in turns)

    def test_static_assets_are_untouched_by_conversation(self, tmp_path: Path):
        """에이전트는 정적 파일을 수정하지 않는다 — 디렉토리 경계가 그것을 드러낸다."""
        first, _ = _make_two_characters(tmp_path)
        layout = CharacterLayout.of(first)
        before = layout.persona_path.read_text(encoding="utf-8")

        _talk(first, MEMORABLE, "반갑소, 박서준.")

        assert layout.persona_path.read_text(encoding="utf-8") == before
