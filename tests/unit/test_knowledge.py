"""KnowledgeModule 단위 테스트 — load_all, getters, to_prompt."""

from __future__ import annotations

import pytest

from src.modules.knowledge import KnowledgeModule


@pytest.fixture
def module(knowledge_dir: str) -> KnowledgeModule:
    """knowledge_dir fixture로 초기화된 KnowledgeModule 인스턴스."""
    m = KnowledgeModule(knowledge_dir)
    m.load_all()
    return m


# ─── load_all ───


def test_load_all_parses_yaml_files(module: KnowledgeModule) -> None:
    """load_all 이후 구조화 데이터가 비어있지 않아야 한다."""
    assert module.get_world() is not None
    assert len(module.get_characters()) > 0 or len(module.get_relationships()) > 0


# ─── get_world ───


def test_get_world_returns_dict(module: KnowledgeModule) -> None:
    """get_world()는 dict를 반환한다."""
    world = module.get_world()
    assert isinstance(world, dict)


def test_get_world_contains_expected_keys(module: KnowledgeModule) -> None:
    """world dict에 name, era 키가 포함된다."""
    world = module.get_world()
    assert world is not None
    assert "name" in world
    assert "era" in world


# ─── get_characters ───


def test_get_characters_returns_list(module: KnowledgeModule) -> None:
    """get_characters()는 list를 반환한다."""
    result = module.get_characters()
    assert isinstance(result, list)


# ─── get_relationships ───


def test_get_relationships_returns_list(module: KnowledgeModule) -> None:
    """get_relationships()는 list를 반환한다."""
    result = module.get_relationships()
    assert isinstance(result, list)


def test_get_relationships_has_items(module: KnowledgeModule) -> None:
    """relationships.yaml이 존재하므로 관계가 최소 1개 이상이다."""
    result = module.get_relationships()
    assert len(result) >= 1


def test_relationship_items_are_dicts(module: KnowledgeModule) -> None:
    """각 관계 항목은 dict 타입이다."""
    for rel in module.get_relationships():
        assert isinstance(rel, dict)


# ─── get_timeline ───


def test_get_timeline_returns_list(module: KnowledgeModule) -> None:
    """get_timeline()는 list를 반환한다."""
    result = module.get_timeline()
    assert isinstance(result, list)


def test_get_timeline_has_items(module: KnowledgeModule) -> None:
    """timeline.yaml이 존재하므로 이벤트가 최소 1개 이상이다."""
    result = module.get_timeline()
    assert len(result) >= 1


# ─── get_locations ───


def test_get_locations_returns_list(module: KnowledgeModule) -> None:
    """get_locations()는 list를 반환한다."""
    result = module.get_locations()
    assert isinstance(result, list)


def test_get_locations_has_items(module: KnowledgeModule) -> None:
    """locations.yaml이 존재하므로 장소가 최소 1개 이상이다."""
    result = module.get_locations()
    assert len(result) >= 1


# ─── to_prompt ───


def test_to_prompt_returns_non_empty_string(module: KnowledgeModule) -> None:
    """to_prompt()는 비어있지 않은 문자열을 반환한다."""
    result = module.to_prompt()
    assert isinstance(result, str)
    assert len(result) > 0


def test_to_prompt_contains_world_info(module: KnowledgeModule) -> None:
    """프롬프트에 세계관 이름이 포함된다."""
    world = module.get_world()
    assert world is not None
    prompt = module.to_prompt()
    assert world.get("name", "") in prompt or "세계관" in prompt
