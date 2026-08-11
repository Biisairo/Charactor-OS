"""FewShotModule 단위 테스트 — load_all, get_all_tags, search, to_prompt."""

from __future__ import annotations

import pytest

from src.modules.fewshot import FewShotModule


@pytest.fixture
def module(examples_dir: str) -> FewShotModule:
    """examples_dir fixture로 초기화된 FewShotModule 인스턴스."""
    m = FewShotModule(examples_dir)
    m.load_all()
    return m


# ─── load_all ───


def test_load_all_parses_yaml_files(module: FewShotModule) -> None:
    """load_all 이후 5개 YAML 파일에서 그룹이 로드된다."""
    groups = module.get_all_groups()
    assert len(groups) == 5


def test_load_all_each_group_has_examples(module: FewShotModule) -> None:
    """각 그룹에 예시가 최소 1개 이상이다."""
    for group in module.get_all_groups():
        assert len(group.examples) >= 1


def test_load_all_examples_have_required_fields(module: FewShotModule) -> None:
    """각 예시에 user, character 필드가 존재한다."""
    for group in module.get_all_groups():
        for ex in group.examples:
            assert hasattr(ex, "user")
            assert hasattr(ex, "character")
            assert isinstance(ex.user, str)
            assert isinstance(ex.character, str)


def test_load_all_resets_before_loading(examples_dir: str) -> None:
    """load_all 호출 시 기존 그룹을 초기화한다."""
    m = FewShotModule(examples_dir)
    m.load_all()
    first_count = len(m.get_all_groups())
    m.load_all()
    assert len(m.get_all_groups()) == first_count


# ─── get_all_tags ───


def test_get_all_tags_returns_list(module: FewShotModule) -> None:
    """get_all_tags()는 list를 반환한다."""
    result = module.get_all_tags()
    assert isinstance(result, list)


def test_get_all_tags_non_empty(module: FewShotModule) -> None:
    """로드 후 태그 목록이 비어있지 않다."""
    result = module.get_all_tags()
    assert len(result) > 0


def test_get_all_tags_contains_expected_tags(module: FewShotModule) -> None:
    """hong-gil-dong 캐릭터의 태그들이 포함된다."""
    tags = module.get_all_tags()
    for expected in ("인사", "유머", "갈등", "위로", "일상"):
        assert expected in tags


# ─── search ───


def test_search_returns_list(module: FewShotModule) -> None:
    """search()는 list를 반환한다."""
    result = module.search("안녕")
    assert isinstance(result, list)


def test_search_matching_tag_returns_results(module: FewShotModule) -> None:
    """태그 키워드가 포함된 쿼리로 검색하면 결과가 반환된다."""
    result = module.search("안녕하세요")
    assert len(result) > 0


def test_search_includes_matching_examples(module: FewShotModule) -> None:
    """검색 결과에 인사 관련 예시가 포함된다."""
    result = module.search("안녕")
    users = [ex.user for ex in result]
    # greeting.yaml의 첫 예시 user가 "안녕!" 이므로 포함되어야 한다
    assert any("안녕" in u for u in users)


def test_search_no_match_returns_empty(module: FewShotModule) -> None:
    """매칭 키워드가 없는 쿼리는 빈 결과를 반환한다."""
    result = module.search("zzzzzzzzzzzzzzzz")
    assert result == []


def test_search_top_k_limits_results(module: FewShotModule) -> None:
    """top_k 파라미터가 결과 수를 제한한다."""
    result = module.search("안녕", top_k=1)
    assert len(result) <= 1


def test_search_respects_top_k_with_emotions(module: FewShotModule) -> None:
    """감정 정보 포함 시에도 top_k가 적용된다."""
    result = module.search("화가 나", emotions={"분노": 0.9}, top_k=2)
    assert len(result) <= 2


def test_search_empty_query_returns_empty(module: FewShotModule) -> None:
    """빈 쿼리는 빈 결과를 반환한다."""
    result = module.search("")
    assert result == []


# ─── to_prompt ───


def test_to_prompt_returns_str(module: FewShotModule) -> None:
    """to_prompt()는 str을 반환한다."""
    result = module.to_prompt("안녕")
    assert isinstance(result, str)


def test_to_prompt_non_empty_for_match(module: FewShotModule) -> None:
    """매칭되는 쿼리에 대해 비어있지 않은 문자열을 반환한다."""
    result = module.to_prompt("안녕하세요")
    assert len(result) > 0


def test_to_prompt_contains_header(module: FewShotModule) -> None:
    """결과에 [예시 대화] 헤더가 포함된다."""
    result = module.to_prompt("안녕하세요")
    assert "[예시 대화]" in result


def test_to_prompt_contains_dialogue_format(module: FewShotModule) -> None:
    """결과에 사용자/캐릭터 대화 형식이 포함된다."""
    result = module.to_prompt("안녕하세요")
    assert "사용자:" in result
    assert "캐릭터:" in result


def test_to_prompt_empty_for_no_match(module: FewShotModule) -> None:
    """매칭 안 되는 쿼리는 빈 문자열을 반환한다."""
    result = module.to_prompt("zzzzzzzzzzzzzzzz")
    assert result == ""


def test_to_prompt_respects_token_budget(module: FewShotModule) -> None:
    """token_budget가 0이면 예시가 포함되지 않는다."""
    result = module.to_prompt("안녕", token_budget=0)
    assert result == ""
