"""KnowledgeModule 단위 테스트 — 로드와 목차.

배경/일반 분리, front matter, 청킹, 검색은 `test_knowledge_rag.py`가 다룬다.
"""

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


def test_load_all_reads_markdown(module: KnowledgeModule) -> None:
    """실제 캐릭터 자산이 검색 가능한 조각으로 올라온다."""
    assert module.chunks()


def test_load_all_reads_era_from_front_matter(module: KnowledgeModule) -> None:
    """era는 Reflection의 시대 기준이 된다. 비면 그 기준이 조용히 꺼진다."""
    assert module.era()


class TestIndexExposesDocumentHeadings:
    """목차는 파일명이 아니라 **무엇이 들어있는지**를 보여야 한다 (SPEC-09 REQ-RA-80).

    파일명만 있으면 뇌는 그 문서에 인삿말 규칙이 있는지 알 수 없어 검색을
    시도조차 하지 않는다. 실사용에서 관측된 결함이다.
    """

    def _module(self, tmp_path, body: str):
        (tmp_path / "broadcast.md").write_text(body, encoding="utf-8")
        module = KnowledgeModule(str(tmp_path))
        module.load_all()
        return module

    def test_headings_appear_in_index(self, tmp_path):
        body = "# 방송 정보\n\n## 스케줄\n주 5회\n\n## 시청자 호칭과 문화\n인사는 쏘하\n"

        index = self._module(tmp_path, body).to_index()

        assert "시청자 호칭과 문화" in index

    def test_body_text_is_not_in_index(self, tmp_path):
        body = "# 방송 정보\n\n## 시청자 호칭과 문화\n방송 시작 인사는 쏘하로 고정되어 있다\n"

        index = self._module(tmp_path, body).to_index()

        assert "고정되어 있다" not in index

    def test_document_without_headings_falls_back_to_filename(self, tmp_path):
        index = self._module(tmp_path, "제목 없는 그냥 본문입니다").to_index()

        assert "broadcast.md" in index

    def test_heading_count_is_capped(self, tmp_path):
        body = "\n".join(f"## 소제목 {i}\n내용\n" for i in range(30))

        index = self._module(tmp_path, body).to_index()

        assert index.count("소제목") <= 8
