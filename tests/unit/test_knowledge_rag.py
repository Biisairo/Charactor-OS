"""배경지식/일반지식 분리와 RAG 색인 (TASK-20).

지식은 성격이 둘이다. 항상 알아야 하는 배경과, 필요할 때 찾아보는 자료.
이 테스트는 그 경계가 디렉토리로 지켜지는지, 그리고 찾아보는 쪽이 표현이
어긋난 질의에도 걸리는지를 본다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.embedding import QUERY
from src.modules.knowledge import (
    EMBEDDING_TOP_K,
    MAX_CHUNK_CHARS,
    NO_KEYWORD_GATE,
    KnowledgeModule,
)

BROADCAST = """# 방송 정보

## 스케줄

주 5회, 화~토 밤 10시 시작.

## 시청자 호칭과 문화

- 시청자를 "얘들아"라고 부른다.
- 방송 시작 인사는 "쏘하", "소하"로 고정되어 있다.
"""


def _fake_embed(text: str, _kind: str) -> np.ndarray:
    """결정론적 더미 임베딩. 같은 낱말을 공유할수록 가까워진다."""
    vec = np.zeros(64, dtype=np.float32)
    for word in set(text.lower().split()):
        vec[hash(word) % 64] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _controlled_embed(similarities: dict[str, float]):
    """조각별 유사도를 지정하는 임베딩 더블.

    질의를 단위원의 (1, 0)에 두고 조각을 (sim, √(1-sim²))에 두면 코사인이
    정확히 `sim`이 된다. 순위 상한과 게이트는 경계값을 재는 테스트이므로
    유사도가 우연에 좌우되면 안 된다.
    """

    def embed(text: str, kind: str) -> np.ndarray:
        if kind == QUERY:
            return np.array([1.0, 0.0], dtype=np.float32)
        similarity = next((s for marker, s in similarities.items() if marker in text), 0.0)
        return np.array([similarity, math.sqrt(max(0.0, 1.0 - similarity**2))], dtype=np.float32)

    return embed


def _write(root: Path, relative: str, body: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _module(root: Path, *, embed=None) -> KnowledgeModule:
    module = KnowledgeModule(str(root), embedding_fn=embed)
    module.load_all()
    return module


# ---------------------------------------------------------------------------
# 1. 디렉토리가 성격을 정한다 (REQ-20-1 · 20-2)
# ---------------------------------------------------------------------------


class TestDirectoriesDecideRole:
    def test_base_document_is_injected_verbatim(self, tmp_path):
        _write(tmp_path, "base/principles.md", "# 원칙\n\n술 방송은 하지 않는다.")

        prompt = _module(tmp_path).to_base_prompt()

        assert "술 방송은 하지 않는다" in prompt

    def test_general_document_is_not_injected(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        prompt = _module(tmp_path).to_base_prompt()

        assert "얘들아" not in prompt

    def test_general_document_is_searchable(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        assert "쏘하" in _module(tmp_path).search_relevant("쏘하")

    def test_base_document_is_not_in_the_search_index(self, tmp_path):
        """이미 프롬프트에 있는 것을 검색으로 또 퍼오면 같은 내용이 두 번 실린다."""
        _write(tmp_path, "base/principles.md", "# 원칙\n\n술 방송은 하지 않는다.")

        assert "술 방송" not in _module(tmp_path).search_relevant("술 방송")

    def test_root_level_file_is_treated_as_general(self, tmp_path):
        """base/ general/ 이 없는 기존 캐릭터가 수정 없이 동작해야 한다."""
        _write(tmp_path, "broadcast.md", BROADCAST)

        module = _module(tmp_path)

        assert "쏘하" in module.search_relevant("쏘하")
        assert "얘들아" not in module.to_base_prompt()

    def test_missing_directory_yields_empty_results(self, tmp_path):
        module = _module(tmp_path)

        assert module.to_base_prompt() == ""
        assert module.search_relevant("아무거나") == ""


# ---------------------------------------------------------------------------
# 2. 배경지식이 인덱스가 된다 (REQ-20-4)
# ---------------------------------------------------------------------------


class TestBasePromptCarriesTheIndex:
    def test_general_headings_are_listed(self, tmp_path):
        _write(tmp_path, "base/principles.md", "# 원칙\n\n술 방송은 하지 않는다.")
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        prompt = _module(tmp_path).to_base_prompt()

        assert "시청자 호칭과 문화" in prompt

    def test_index_appears_even_without_base_documents(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        assert "시청자 호칭과 문화" in _module(tmp_path).to_base_prompt()

    def test_index_does_not_leak_body_text(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        assert "얘들아" not in _module(tmp_path).to_base_prompt()

    def test_base_text_precedes_the_index(self, tmp_path):
        _write(tmp_path, "base/principles.md", "# 원칙\n\n술 방송은 하지 않는다.")
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        prompt = _module(tmp_path).to_base_prompt()

        assert prompt.index("술 방송") < prompt.index("시청자 호칭과 문화")


# ---------------------------------------------------------------------------
# 4. 청킹 (REQ-20-5 · 20-6)
# ---------------------------------------------------------------------------


class TestChunking:
    def test_sections_split_on_headings(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        chunks = _module(tmp_path).chunks()

        assert len(chunks) >= 2

    def test_chunk_keeps_its_heading_path(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        chunks = _module(tmp_path).chunks()
        greeting = next(c for c in chunks if "쏘하" in c.text)

        assert "방송 정보" in greeting.heading_path
        assert "시청자 호칭과 문화" in greeting.heading_path

    def test_long_section_is_split_further_by_paragraph(self, tmp_path):
        paragraphs = "\n\n".join(f"{i}번째 문단입니다. " * 12 for i in range(8))
        _write(tmp_path, "general/long.md", f"# 긴 문서\n\n## 큰 절\n\n{paragraphs}")

        chunks = _module(tmp_path).chunks()

        assert len(chunks) > 1
        assert all(len(c.text) <= MAX_CHUNK_CHARS * 2 for c in chunks)

    def test_short_section_stays_whole(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        chunks = _module(tmp_path).chunks()
        schedule = next(c for c in chunks if "화~토" in c.text)

        assert "주 5회" in schedule.text

    def test_document_without_headings_becomes_one_chunk(self, tmp_path):
        _write(tmp_path, "general/plain.md", "제목 없는 짧은 본문입니다.")

        assert len(_module(tmp_path).chunks()) == 1

    def test_search_result_shows_heading_path(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        assert "시청자 호칭과 문화" in _module(tmp_path).search_relevant("쏘하")


# ---------------------------------------------------------------------------
# 5. 하이브리드 검색 (REQ-20-7)
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def test_keyword_hit_without_embeddings(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        assert "쏘하" in _module(tmp_path).search_relevant("쏘하")

    def test_substring_query_still_hits(self, tmp_path):
        """실사용 실패 사례 — 사용자가 "쏘하쏘하하"라고 쳤다."""
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        assert "쏘하" in _module(tmp_path).search_relevant("쏘하쏘하하")

    def test_embedding_finds_reworded_query(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        module = _module(tmp_path, embed=_fake_embed)

        assert module.search_relevant("시청자 호칭과 문화") != ""

    def test_unrelated_query_returns_nothing(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        assert _module(tmp_path).search_relevant("양자역학 강의 노트") == ""

    def test_token_budget_is_respected(self, tmp_path):
        body = "\n\n".join(f"## 절 {i}\n\n쏘하 관련 내용 {i}. " * 6 for i in range(20))
        _write(tmp_path, "general/big.md", f"# 큰 문서\n\n{body}")

        result = _module(tmp_path).search_relevant("쏘하", token_budget=120)

        assert KnowledgeModule._estimate_tokens(result) <= 120 * 1.2

    def test_broken_embedding_degrades_to_keywords(self, tmp_path):
        """임베딩이 죽어도 검색이 조용히 0건이 되면 안 된다 (REQ-15-1과 같은 원칙)."""

        def _broken(_text, _kind):
            raise RuntimeError("모델 없음")

        _write(tmp_path, "general/broadcast.md", BROADCAST)

        module = _module(tmp_path, embed=_broken)

        assert "쏘하" in module.search_relevant("쏘하")
        assert any(not issue.expected for issue in module.load_issues)


# ---------------------------------------------------------------------------
# 5.1 순위 상한과 무키워드 게이트 (SPEC-12 REQ-21-6 ~ 21-8)
#
# 절대 임계값을 폐기한 자리다. 새 모델은 유사도가 0.79~0.86에 압축되어 적중과
# 잡음의 분포가 겹치므로, 어떤 임계값으로도 "관련된 것이 없다"를 판정할 수
# 없다 (SPEC-12 4.2). 대신 모델이 잘 하는 것 — 순위 — 를 쓴다.
# ---------------------------------------------------------------------------

THREE_SECTIONS = """# 자료

## 첫째 절

가장 비슷한 내용.

## 둘째 절

두 번째로 비슷한 내용.

## 셋째 절

세 번째로 비슷한 내용.
"""


class TestRankAndGate:
    def _module_with(self, tmp_path, similarities: dict[str, float]) -> KnowledgeModule:
        _write(tmp_path, "general/three.md", THREE_SECTIONS)
        return _module(tmp_path, embed=_controlled_embed(similarities))

    def test_gate_blocks_chunk_without_keyword(self, tmp_path):
        """키워드 근거가 없는 조각은 게이트를 넘지 못하면 후보가 아니다."""
        module = self._module_with(tmp_path, {"첫째": NO_KEYWORD_GATE - 0.01})

        assert module.search_relevant("전혀 다른 표현") == ""

    def test_gate_admits_chunk_above_threshold(self, tmp_path):
        """의역 질의는 게이트를 넘겨 통과한다 — 임베딩을 붙인 이유다."""
        module = self._module_with(tmp_path, {"첫째": NO_KEYWORD_GATE + 0.01})

        assert "가장 비슷한 내용" in module.search_relevant("전혀 다른 표현")

    def test_keyword_hit_is_exempt_from_gate(self, tmp_path):
        """키워드가 걸린 조각에는 게이트를 적용하지 않는다.

        이미 다른 근거가 있으므로 임베딩은 순위만 조정하면 된다.
        """
        module = self._module_with(tmp_path, {"첫째": 0.1})

        assert "가장 비슷한 내용" in module.search_relevant("첫째 절")

    def test_only_top_k_receive_bonus(self, tmp_path):
        """상위 K개 밖은 유사도가 게이트를 넘어도 후보가 되지 않는다."""
        module = self._module_with(tmp_path, {"첫째": 0.99, "둘째": 0.98, "셋째": 0.97})

        result = module.search_relevant("전혀 다른 표현")

        assert "세 번째로 비슷한 내용" not in result
        assert EMBEDDING_TOP_K == 2

    def test_top_k_are_all_returned(self, tmp_path):
        module = self._module_with(tmp_path, {"첫째": 0.99, "둘째": 0.98, "셋째": 0.97})

        result = module.search_relevant("전혀 다른 표현")

        assert "가장 비슷한 내용" in result
        assert "두 번째로 비슷한 내용" in result

    def test_embedding_orders_results(self, tmp_path):
        """키워드가 같은 두 조각의 순서는 임베딩이 정한다."""
        module = self._module_with(tmp_path, {"둘째": 0.99, "첫째": 0.5})

        result = module.search_relevant("비슷한 내용")

        assert result.index("두 번째로") < result.index("가장 비슷한")

    def test_no_embedding_uses_keywords_only(self, tmp_path):
        """`embedding_fn`이 없으면 키워드 점수만으로 돈다 (기존 정책 유지)."""
        _write(tmp_path, "general/three.md", THREE_SECTIONS)

        assert "가장 비슷한 내용" in _module(tmp_path).search_relevant("첫째 절")


# ---------------------------------------------------------------------------
# 6. 배경지식 총량 경고 (REQ-20-8)
# ---------------------------------------------------------------------------


class TestBaseSizeWarning:
    def test_oversized_base_reports_an_issue(self, tmp_path):
        _write(tmp_path, "base/huge.md", "# 배경\n\n" + "가나다라마바사 " * 4000)

        module = _module(tmp_path)

        assert any("배경" in issue.reason or "base" in issue.reason for issue in module.load_issues)

    def test_oversized_base_is_not_truncated(self, tmp_path):
        """사람이 넣은 배경을 임의로 자르면 캐릭터가 무너진다."""
        body = "가나다라마바사 " * 4000
        _write(tmp_path, "base/huge.md", f"# 배경\n\n{body}")

        assert body.strip() in _module(tmp_path).to_base_prompt()

    def test_normal_base_reports_nothing(self, tmp_path):
        _write(tmp_path, "base/principles.md", "# 원칙\n\n술 방송은 하지 않는다.")

        assert _module(tmp_path).load_issues == []


# ---------------------------------------------------------------------------
# 7. 마크다운 단일 형식 (TASK-22)
# ---------------------------------------------------------------------------


FRONT_MATTER_DOC = """---
era: 2020년대 후반 대한민국 서울
---

# 이 바닥이 돌아가는 방식

클립은 영구히 남는다.
"""


class TestFrontMatter:
    def test_era_is_read(self, tmp_path):
        _write(tmp_path, "base/01-world.md", FRONT_MATTER_DOC)

        assert _module(tmp_path).era() == "2020년대 후반 대한민국 서울"

    def test_front_matter_is_not_in_the_prompt(self, tmp_path):
        """메타데이터는 본문이 아니다. 프롬프트에 새면 노이즈가 된다."""
        _write(tmp_path, "base/01-world.md", FRONT_MATTER_DOC)

        prompt = _module(tmp_path).to_base_prompt()

        assert "클립은 영구히 남는다" in prompt
        assert "era:" not in prompt
        assert "---" not in prompt

    def test_front_matter_is_not_indexed(self, tmp_path):
        _write(tmp_path, "general/world.md", FRONT_MATTER_DOC)

        module = _module(tmp_path)

        assert all("era:" not in c.text for c in module.chunks())

    def test_era_is_read_from_general_too(self, tmp_path):
        _write(tmp_path, "general/world.md", FRONT_MATTER_DOC)

        assert _module(tmp_path).era() == "2020년대 후반 대한민국 서울"

    def test_missing_front_matter_yields_empty_era(self, tmp_path):
        _write(tmp_path, "base/01-world.md", "# 제목\n\n본문")

        assert _module(tmp_path).era() == ""

    def test_first_file_wins_when_several_declare_era(self, tmp_path):
        _write(tmp_path, "base/01-world.md", "---\nera: 첫 번째\n---\n\n# 하나")
        _write(tmp_path, "base/02-other.md", "---\nera: 두 번째\n---\n\n# 둘")

        assert _module(tmp_path).era() == "첫 번째"

    def test_broken_front_matter_is_reported(self, tmp_path):
        _write(tmp_path, "base/01-world.md", "---\nera: [닫히지 않음\n---\n\n# 제목\n\n본문")

        module = _module(tmp_path)

        assert module.era() == ""
        assert any(not issue.expected for issue in module.load_issues)
        assert "본문" in module.to_base_prompt(), "메타가 깨져도 본문은 살아야 한다"


class TestMarkdownOnly:
    def test_yaml_files_are_ignored(self, tmp_path):
        _write(tmp_path, "general/world.yaml", 'type: world\nera: "조선"\n')

        module = _module(tmp_path)

        assert module.chunks() == []
        assert module.era() == ""

    def test_markdown_is_read(self, tmp_path):
        _write(tmp_path, "general/broadcast.md", BROADCAST)

        assert _module(tmp_path).chunks()

    def test_text_file_is_ignored(self, tmp_path):
        _write(tmp_path, "general/notes.txt", "메모")

        assert _module(tmp_path).chunks() == []
