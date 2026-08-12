"""정적 자산 로드 실패의 가시성 (TASK-06, REQ-06-1 ~ REQ-06-4).

`FewShotModule`은 예시 파일 파싱 실패를 `except Exception: pass`로 무시했다.
YAML 문법 오류가 있으면 few-shot 예시가 조용히 0개가 되고, 프롬프트 품질이
원인 불명으로 떨어진다. 사용자도 개발자도 이를 인지할 수 없다.

`KnowledgeModule`의 같은 패턴은 **의도된 폴백**(구조화 실패 → freeform 처리)이므로
구별되어야 한다 (REQ-06-3).

이 테스트는 LLM API 키 없이 결정론적으로 통과한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.modules.fewshot import FewShotModule
from src.modules.knowledge import KnowledgeModule

BROKEN_YAML = "tag: greeting\nexamples:\n  - user: '따옴표가 안 닫힘\n    character: 반갑소\n"

VALID_YAML = """tag: comfort
examples:
  - user: 오늘 힘들었어
    character: 그리 상심 말게.
"""


@pytest.fixture
def examples_dir(tmp_path: Path) -> Path:
    d = tmp_path / "examples"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# REQ-06-1 · REQ-06-2 — 예시 로드 실패가 기록되고, 나머지는 계속 로드된다
# ---------------------------------------------------------------------------


class TestFewShotLoadIssues:
    def test_broken_file_is_recorded(self, examples_dir: Path):
        (examples_dir / "greeting.yaml").write_text(BROKEN_YAML, encoding="utf-8")
        module = FewShotModule(str(examples_dir))
        module.load_all()

        assert len(module.load_issues) == 1
        issue = module.load_issues[0]
        assert issue.filename == "greeting.yaml"
        assert issue.reason, "원인이 비어 있으면 디버깅에 쓸모가 없다"

    def test_broken_file_does_not_block_valid_files(self, examples_dir: Path):
        (examples_dir / "greeting.yaml").write_text(BROKEN_YAML, encoding="utf-8")
        (examples_dir / "comfort.yaml").write_text(VALID_YAML, encoding="utf-8")
        module = FewShotModule(str(examples_dir))
        module.load_all()

        assert module.get_all_tags() == ["comfort"]
        assert len(module.load_issues) == 1

    def test_clean_load_records_nothing(self, examples_dir: Path):
        (examples_dir / "comfort.yaml").write_text(VALID_YAML, encoding="utf-8")
        module = FewShotModule(str(examples_dir))
        module.load_all()

        assert module.load_issues == []

    def test_issues_are_reset_on_reload(self, examples_dir: Path):
        (examples_dir / "greeting.yaml").write_text(BROKEN_YAML, encoding="utf-8")
        module = FewShotModule(str(examples_dir))
        module.load_all()
        (examples_dir / "greeting.yaml").write_text(VALID_YAML, encoding="utf-8")
        module.load_all()

        assert module.load_issues == []

    def test_failure_is_marked_unexpected(self, examples_dir: Path):
        (examples_dir / "greeting.yaml").write_text(BROKEN_YAML, encoding="utf-8")
        module = FewShotModule(str(examples_dir))
        module.load_all()

        assert module.load_issues[0].expected is False


# ---------------------------------------------------------------------------
# REQ-06-3 — 의도된 폴백과 예기치 않은 실패가 구별된다
# ---------------------------------------------------------------------------


class TestKnowledgeFallbackIsDistinguishable:
    def test_structured_parse_failure_is_marked_expected(self, tmp_path: Path):
        """구조화 실패는 freeform 폴백이라는 의도된 동작이다."""
        d = tmp_path / "knowledge"
        d.mkdir()
        (d / "notes.yaml").write_text(BROKEN_YAML, encoding="utf-8")

        module = KnowledgeModule(str(d))
        module.load_all()

        assert len(module.load_issues) == 1
        assert module.load_issues[0].expected is True, (
            "freeform 폴백은 의도된 동작이므로 예기치 않은 실패와 구별되어야 한다"
        )

    def test_fallback_still_loads_content_as_freeform(self, tmp_path: Path):
        d = tmp_path / "knowledge"
        d.mkdir()
        (d / "notes.yaml").write_text(BROKEN_YAML, encoding="utf-8")

        module = KnowledgeModule(str(d))
        module.load_all()

        assert module._freeform, "폴백이면 내용은 freeform으로 남아야 한다"

    def test_structured_file_records_nothing(self, tmp_path: Path):
        d = tmp_path / "knowledge"
        d.mkdir()
        (d / "world.yaml").write_text("type: world\nname: 조선\nera: 조선 중기\n", encoding="utf-8")

        module = KnowledgeModule(str(d))
        module.load_all()

        assert module.load_issues == []


# ---------------------------------------------------------------------------
# REQ-06-1 — CharacterOS가 로드 문제를 로그로 올린다
# ---------------------------------------------------------------------------


class TestCharacterOSSurfacesIssues:
    def test_load_issue_appears_in_logs(self, character_dir: Path, tmp_path: Path):
        from tests.conftest import MockClient, make_character_os

        (character_dir / "examples" / "broken.yaml").write_text(BROKEN_YAML, encoding="utf-8")
        cos = make_character_os(character_dir, tmp_path / "state", MockClient())

        joined = "\n".join(cos._debug_logs)
        assert "broken.yaml" in joined, "로드 실패가 로그에 파일명과 함께 남아야 한다"

    def test_expected_fallback_is_labelled_differently(self, character_dir: Path, tmp_path: Path):
        from tests.conftest import MockClient, make_character_os

        (character_dir / "knowledge" / "notes.yaml").write_text(BROKEN_YAML, encoding="utf-8")
        cos = make_character_os(character_dir, tmp_path / "state", MockClient())

        lines = [line for line in cos._debug_logs if "notes.yaml" in line]
        assert lines, "폴백도 기록은 되어야 한다"
        assert any("폴백" in line for line in lines), (
            "의도된 폴백은 실패와 다른 말로 기록되어야 한다 (REQ-06-3)"
        )
