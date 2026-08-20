"""임베딩 모델 (SPEC-12).

Memory·Knowledge·FewShot 이 공유하는 단일 신호다. 세 모듈이 같은 좌표계를
써야 하므로 모델을 여기 한 곳에서 정하고, **무엇으로 만든 벡터인지**를
`model_id`로 드러낸다 — 좌표계가 다른 벡터를 섞으면 오류 없이 무의미한
점수가 나온다 (SPEC-12 P-7).

이 모듈에는 LLM 호출이 없다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

# 한국어를 다루는 모델. `all-MiniLM-L6-v2`는 영어 전용이라 한국어에서 변별력이
# **음수**였다 — 무관한 질의가 관련 질의보다 높은 유사도를 받았다
# (SPEC-12 4.2, top-1 0%).
DEFAULT_MODEL = "intfloat/multilingual-e5-small"

QUERY: Literal["query"] = "query"
PASSAGE: Literal["passage"] = "passage"

EmbeddingKind = Literal["query", "passage"]


@dataclass(frozen=True)
class EmbeddingConfig:
    """모델과 그 모델이 요구하는 규약.

    프리픽스가 설정에 있는 이유는 모델마다 규약이 다르기 때문이다. 코드에
    박으면 모델 교체가 다시 코드 변경이 된다 (SPEC-12 결정 6).

    기본값이 빈 문자열인 것은 실측 결과다. e5 계열은 `query:`/`passage:`를
    권하지만, 이 자산에서는 프리픽스를 붙인 쪽이 top-1 이 6%p 낮았다
    (SPEC-12 4.1).
    """

    model: str = DEFAULT_MODEL
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True


class Embedder:
    """텍스트를 벡터로 만든다. 용도(`kind`)를 반드시 받는다.

    `kind`에 기본값을 두지 않는 것이 이 클래스의 유일한 강제다. 기본값이
    있으면 호출부가 용도를 생각하지 않고 지나가며, 그 실수는 검색 품질
    저하로만 나타나 추적할 수 없다 (SPEC-12 5.2).
    """

    # 모델 로드는 수 초가 걸린다. 식별자별로 한 번만 만든다.
    _models: dict[str, object] = {}

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        encode: Callable[..., np.ndarray] | None = None,
    ):
        """
        Args:
            encode: 인코딩 함수. 주입하면 모델을 로드하지 않는다 — 테스트가
                수백 MB 로드 없이 프리픽스와 용도 구분을 검증하는 경로다.
        """
        self._config = config or EmbeddingConfig()
        self._encode = encode

    @property
    def model_id(self) -> str:
        """이 임베더가 만든 벡터의 좌표계 식별자."""
        return self._config.model

    def __call__(self, text: str, kind: EmbeddingKind) -> np.ndarray:
        if kind == QUERY:
            prefix = self._config.query_prefix
        elif kind == PASSAGE:
            prefix = self._config.passage_prefix
        else:
            raise ValueError(f"알 수 없는 임베딩 용도: {kind!r} — {QUERY} 또는 {PASSAGE}")

        return self._encoder()(f"{prefix}{text}", normalize_embeddings=self._config.normalize)

    def _encoder(self) -> Callable[..., np.ndarray]:
        if self._encode is not None:
            return self._encode

        model = Embedder._models.get(self.model_id)
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_id)
            Embedder._models[self.model_id] = model

        return model.encode


def from_config(config: dict) -> Embedder:
    """`config.yaml`의 `embedding` 섹션으로 임베더를 만든다 (REQ-21-2).

    섹션이 없는 것은 오류가 아니다 — 기본값으로 돈다.
    """
    section = config.get("embedding") or {}
    defaults = EmbeddingConfig()
    return Embedder(
        EmbeddingConfig(
            model=str(section.get("model") or defaults.model),
            query_prefix=str(section.get("query_prefix") or ""),
            passage_prefix=str(section.get("passage_prefix") or ""),
            normalize=bool(section.get("normalize", defaults.normalize)),
        )
    )
