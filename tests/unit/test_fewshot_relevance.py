"""few-shot 관련성 임계값과 캐릭터 고유 어휘 (TASK-15 / REQ-15-1, REQ-15-3).

두 가지를 고정한다.

1. 관련성이 낮으면 **아무것도 반환하지 않는다.** 예전에는 점수가 0만 아니면
   무엇이든 반환했고, "광합성의 원리를 설명해줘" 같은 질의에도 갈등 예시가
   프롬프트에 들어갔다. 무관한 예시는 응답 품질을 조용히 떨어뜨린다.

2. 태그 트리거 어휘를 **캐릭터 자산에서 확장**할 수 있다. 내장 어휘만 쓰면
   캐릭터 고유 어휘(`도네`·`관군`)에서 태그 점수가 0이 되고 판단이 통째로
   임베딩으로 넘어간다.

임베딩을 테스트가 지배하므로 결정론적이다. API 키가 필요 없다.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from src.modules.fewshot import MIN_FEWSHOT_SCORE, FewShotModule

# 질의와 예시의 유사도를 테스트가 직접 정한다.
_SIMILARITY: dict[str, float] = {}

# 현재 임베딩 모델의 **실측** 유사도 대역 (SPEC-12 4.2 · 4.5).
#
# 이 두 상수가 이 파일의 핵심이다. 예전에는 잡음을 0.5로 박아두었고, 그것이
# 낡은 모델(`all-MiniLM-L6-v2`)의 분포였다. 모델을 바꾸자 무관한 질의가 0.84를
# 받게 되면서 임계값이 무력해졌는데 **테스트는 계속 통과했다** (SPEC-12 P-13).
#
# 모델을 바꾸면 이 값을 `eval/embedding_probe.py`로 다시 재야 한다. 값이 실제
# 분포와 어긋나면 이 파일의 테스트는 아무것도 지키지 못한다.
NOISE_SIMILARITY = 0.84
HIT_SIMILARITY = 0.95


def _embedding(text: str, _kind: str = "passage") -> np.ndarray:
    """단위 벡터를 만들되, 지정된 유사도가 나오도록 각도를 잡는다."""
    similarity = _SIMILARITY.get(text, 0.0)
    return np.array([similarity, np.sqrt(max(0.0, 1.0 - similarity**2))], dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_similarity():
    _SIMILARITY.clear()
    _SIMILARITY["__query__"] = 1.0  # 질의는 기준축
    yield
    _SIMILARITY.clear()


def _write_group(directory, tag: str, user: str, keywords: list[str] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict = {"tag": tag, "examples": [{"user": user, "character": f"{tag} 응답"}]}
    if keywords is not None:
        payload["keywords"] = keywords
    (directory / f"{tag}.yaml").write_text(yaml.dump(payload, allow_unicode=True), encoding="utf-8")


def _module(directory, *, with_embedding: bool = True) -> FewShotModule:
    module = FewShotModule(str(directory), embedding_fn=_embedding if with_embedding else None)
    module.load_all()
    return module


# ---------------------------------------------------------------------------
# REQ-15-3 — 관련성이 임계값 미만이면 예시를 넣지 않는다
# ---------------------------------------------------------------------------


class TestRelevanceThreshold:
    def test_noise_band_currently_passes_the_threshold(self, tmp_path):
        """**현재 임계값은 무관 대역을 막지 못한다.** 그 사실을 고정한다.

        예전 이 테스트는 잡음을 0.5로 박아두고 "무관하면 걸러진다"를 주장했다.
        낡은 모델의 분포였고, 모델을 바꾼 뒤로는 사실이 아니다.

        임계를 올려 막아 봤지만 실제 평가 점수가 4.867 → 4.550 으로 떨어졌다 —
        평가 질의 40건 중 11건이 예시를 잃었기 때문이다 (SPEC-12 4.5). 무관한
        예시보다 **예시가 없는 것이 더 해롭다**는 것이 실측이다.

        그래서 이 테스트는 "막힌다"가 아니라 **막히지 않는다는 현재 상태**를
        적어 둔다. 주장과 실제가 어긋난 테스트가 통과하는 것이 더 나쁘다.
        """
        _write_group(tmp_path, "인사", "안녕")
        _SIMILARITY["안녕"] = NOISE_SIMILARITY

        assert _module(tmp_path).search("__query__") != []

    def test_hit_band_similarity_still_returns(self, tmp_path):
        _write_group(tmp_path, "인사", "안녕")
        _SIMILARITY["안녕"] = HIT_SIMILARITY

        assert len(_module(tmp_path).search("__query__")) == 1

    def test_hit_band_clears_the_threshold(self, tmp_path):
        """적중 대역은 임계를 넘어야 한다. 이쪽은 반드시 지켜져야 하는 방향이다.

        잡음 대역까지 함께 막으려 하면 실제 질의가 예시를 잃는다 (SPEC-12 4.5).
        """
        from src.modules.fewshot import EMBEDDING_WEIGHT

        assert HIT_SIMILARITY * EMBEDDING_WEIGHT >= MIN_FEWSHOT_SCORE

    def test_to_prompt_is_empty_when_score_is_truly_low(self, tmp_path):
        """점수가 임계 아래면 프롬프트에 예시 블록이 붙지 않는다.

        임베딩이 잡음 대역까지 밀어 올리지 못하는 경우 — 예시 임베딩이 없는
        경로(임베딩 실패)나 유사도가 실제로 낮은 경우가 이에 해당한다.
        """
        _write_group(tmp_path, "인사", "안녕")
        _SIMILARITY["안녕"] = 0.1

        assert _module(tmp_path).to_prompt("__query__") == ""

    def test_tag_hit_survives_the_threshold(self, tmp_path):
        """태그가 걸린 정상 질의는 임계값에 걸리지 않아야 한다.

        임계값을 올리면 무관한 예시를 막는 대신 정상 질의를 자를 위험이 생긴다.
        """
        _write_group(tmp_path, "인사", "안녕")
        _SIMILARITY["안녕"] = NOISE_SIMILARITY

        assert _module(tmp_path).search("안녕 반가워")

    def test_fallback_without_embedding_is_not_further_degraded(self, tmp_path):
        """임계값은 임베딩이 있는 점수 체계에서 보정했다.

        폴백(태그 0.7 + 감정 0.3)에 같은 값을 적용하면 키워드 하나만 걸리는
        정상 질의까지 잘려 few-shot이 통째로 빈다. 이미 퇴화한 경로를 더
        깎지 않는다.
        """
        _write_group(tmp_path, "인사", "안녕")

        assert _module(tmp_path, with_embedding=False).search("안녕 반가워")


# ---------------------------------------------------------------------------
# REQ-15-1 — 태그 어휘를 캐릭터 자산에서 확장할 수 있다
# ---------------------------------------------------------------------------


class TestCharacterKeywords:
    def test_keywords_are_loaded_from_the_example_file(self, tmp_path):
        _write_group(tmp_path, "일상", "뭐해?", keywords=["도네", "동접"])

        group = _module(tmp_path).get_all_groups()[0]

        assert group.keywords == ["도네", "동접"]

    def test_custom_keyword_lifts_a_domain_query_over_the_threshold(self, tmp_path):
        """내장 어휘 밖의 고유 어휘로도 예시가 검색된다."""
        _write_group(tmp_path, "일상", "뭐해?", keywords=["도네"])
        _SIMILARITY["뭐해?"] = 0.4  # 임베딩만으로는 0.16 — 임계값 미달

        module = _module(tmp_path)

        assert module.search("__query__") == [], "전제: 임베딩만으로는 미달이어야 한다"
        assert module.search("도네 많이 들어왔어? __query__"), (
            "고유 어휘가 태그 점수를 올려 임계값을 넘겨야 한다"
        )

    def test_adding_keywords_never_lowers_the_score(self, tmp_path):
        """어휘를 늘렸다고 점수가 떨어지면 안 된다.

        예전 태그 점수는 `matches / len(keywords)`였다. 어휘를 추가하면 분모가
        커져 **점수가 오히려 낮아졌다.** 고유 어휘를 더하는 것이 역효과를 내는
        구조였고, 이 테스트가 그 회귀를 막는다.
        """
        bare = tmp_path / "bare"
        extended = tmp_path / "extended"
        _write_group(bare, "일상", "뭐해?")
        _write_group(extended, "일상", "뭐해?", keywords=["도네", "방송", "정산", "합방"])
        _SIMILARITY["뭐해?"] = 0.0

        query = "오늘 뭐해?"
        bare_group = _module(bare).get_all_groups()[0]
        extended_group = _module(extended).get_all_groups()[0]

        bare_score = _module(bare)._tag_score(bare_group, query)
        extended_score = _module(extended)._tag_score(extended_group, query)

        assert extended_score >= bare_score


def test_threshold_is_a_named_constant():
    """임계값이 매직 넘버로 흩어지지 않는다."""
    assert 0.0 < MIN_FEWSHOT_SCORE < 1.0
