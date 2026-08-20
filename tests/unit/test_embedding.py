"""임베딩 모델 설정과 용도 구분 (SPEC-12 REQ-21-1 ~ 21-4).

임베딩은 세 모듈이 공유하는 단일 신호다. 어떤 모델로 만들었는지, 찾는 쪽인지
찾히는 쪽인지가 흐려지면 검색 품질이 원인 불명으로 떨어진다.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embedding import (
    DEFAULT_MODEL,
    PASSAGE,
    QUERY,
    Embedder,
    EmbeddingConfig,
    from_config,
)


def _recording_encode(seen: list[str]):
    """인자로 받은 텍스트를 기록하는 encode 더블."""

    def encode(text: str, normalize_embeddings: bool = True) -> np.ndarray:
        seen.append(text)
        vec = np.ones(4, dtype=np.float32)
        return vec / np.linalg.norm(vec) if normalize_embeddings else vec

    return encode


# ---------------------------------------------------------------------------
# 1. 설정 (REQ-21-2)
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_model_is_multilingual(self):
        """기본 모델은 한국어를 다루는 모델이다 (4.2)."""
        assert from_config({}).model_id == DEFAULT_MODEL

    def test_model_comes_from_config(self):
        embedder = from_config({"embedding": {"model": "some/other-model"}})

        assert embedder.model_id == "some/other-model"

    def test_missing_section_falls_back_to_defaults(self):
        """섹션이 없는 것은 오류가 아니다 — 기본값으로 돈다."""
        assert from_config({}).model_id == from_config({"embedding": {}}).model_id

    def test_default_prefixes_are_empty(self):
        """프리픽스는 기본으로 붙지 않는다 (4.1 — 실측에서 top-1이 낮았다)."""
        config = EmbeddingConfig()

        assert config.query_prefix == ""
        assert config.passage_prefix == ""


# ---------------------------------------------------------------------------
# 2. 용도 구분 (REQ-21-3)
# ---------------------------------------------------------------------------


class TestKind:
    def test_query_prefix_is_applied(self):
        seen: list[str] = []
        embedder = Embedder(
            EmbeddingConfig(query_prefix="query: ", passage_prefix="passage: "),
            encode=_recording_encode(seen),
        )

        embedder("쏘하", QUERY)

        assert seen == ["query: 쏘하"]

    def test_passage_prefix_is_applied(self):
        seen: list[str] = []
        embedder = Embedder(
            EmbeddingConfig(query_prefix="query: ", passage_prefix="passage: "),
            encode=_recording_encode(seen),
        )

        embedder("인삿말은 쏘하다", PASSAGE)

        assert seen == ["passage: 인삿말은 쏘하다"]

    def test_empty_prefix_adds_nothing(self):
        """프리픽스를 쓰지 않는 모델에서 원문이 그대로 가야 한다."""
        seen: list[str] = []
        embedder = Embedder(EmbeddingConfig(), encode=_recording_encode(seen))

        embedder("쏘하", QUERY)

        assert seen == ["쏘하"]

    def test_unknown_kind_is_rejected(self):
        """용도를 틀리면 검색 품질로만 드러난다. 호출 시점에 막는다."""
        embedder = Embedder(EmbeddingConfig(), encode=_recording_encode([]))

        with pytest.raises(ValueError):
            embedder("쏘하", "document")

    def test_kind_has_no_default(self):
        """기본값이 있으면 호출부가 용도를 생각하지 않고 지나간다 (5.2)."""
        embedder = Embedder(EmbeddingConfig(), encode=_recording_encode([]))

        with pytest.raises(TypeError):
            embedder("쏘하")


# ---------------------------------------------------------------------------
# 3. 정규화 (REQ-21-1)
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_normalization_is_requested_by_default(self):
        """Memory 검색이 정규화 벡터를 전제한 내적이다 (P-7)."""
        received: dict = {}

        def encode(text: str, normalize_embeddings: bool = True):
            received["normalize"] = normalize_embeddings
            return np.ones(4, dtype=np.float32)

        Embedder(EmbeddingConfig(), encode=encode)("쏘하", QUERY)

        assert received["normalize"] is True
