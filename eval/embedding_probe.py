"""임베딩 모델의 한국어 변별력 측정 (SPEC-12 REQ-21-5 · REQ-21-6).

    PYTHONPATH=. uv run python -m eval.embedding_probe
    PYTHONPATH=. uv run python -m eval.embedding_probe --json out.json

모델을 내려받아 로컬에서 돈다. LLM API 키가 필요 없다. 기본 테스트 스위트에는
넣지 않는다 — 모델 로드에 수십 초가 걸리고, 변별력은 합격/불합격이 아니라
**상수를 정하기 위한 수치**이기 때문이다.

무엇을 재는가:

    적중 유사도   질의와 그 답이 실린 조각의 유사도
    잡음 유사도   자료에 답이 없는 질의가 끌어오는 최고 유사도
    변별력        둘의 차이. 이것이 작으면 임계값을 어디에 두어도 걸러지지 않는다
    top-1 정확도  임베딩 **단독으로** 정답 조각을 1위로 올리는가

절대 임계값을 쓰려면 잡음 최대와 적중 최소 사이에 구간이 있어야 한다. 그
구간이 비어 있으면(여유가 음수) 임계값으로는 "관련된 것이 없다"를 판정할 수
없다는 뜻이고, 실제로 그랬다 — 그래서 `--hybrid` 가 순위 상한과 무키워드
게이트를 비교한다 (SPEC-12 4.2).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.character_layout import CharacterLayout
from src.modules.knowledge import KnowledgeChunk, KnowledgeModule

PROBE_PATH = Path(__file__).parent / "datasets" / "embedding_probe.yaml"
CHARACTERS_DIR = Path("characters")

UNRELATED = "unrelated"

# 재는 대상. (표시명, 모델 식별자, 질의 프리픽스, 문서 프리픽스)
VARIANTS = [
    ("MiniLM(현재)", "all-MiniLM-L6-v2", "", ""),
    ("e5-small", "intfloat/multilingual-e5-small", "", ""),
    ("e5-small+prefix", "intfloat/multilingual-e5-small", "query: ", "passage: "),
]

THRESHOLDS = [0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]


@dataclass(frozen=True)
class Case:
    character: str
    query: str
    expected: tuple[str, ...]
    band: str

    @property
    def is_unrelated(self) -> bool:
        return self.band == UNRELATED


@dataclass
class Outcome:
    case: Case
    similarity: float  # 적중 유사도, unrelated 는 잡음 유사도
    top1_heading: str
    top1_similarity: float
    top1_correct: bool


def load_cases(path: Path = PROBE_PATH) -> list[Case]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Case(
            character=c["character"],
            query=c["query"],
            expected=tuple(c.get("expected") or ()),
            band=c.get("band", "?"),
        )
        for c in payload["cases"]
    ]


def load_chunks(character: str) -> list[KnowledgeChunk]:
    """청킹만 한다 — 임베딩은 변형별로 따로 계산한다."""
    layout = CharacterLayout.of(CHARACTERS_DIR / character)
    module = KnowledgeModule(str(layout.knowledge_dir))
    module.load_all()
    return module.chunks()


def _matches(chunk: KnowledgeChunk, expected: tuple[str, ...]) -> bool:
    return any(e in chunk.heading_path for e in expected)


def run_variant(
    cases: list[Case], model_id: str, query_prefix: str, passage_prefix: str
) -> list[Outcome]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, normalize_embeddings=True)

    outcomes: list[Outcome] = []
    for character in sorted({c.character for c in cases}):
        chunks = load_chunks(character)
        vectors = encode([f"{passage_prefix}{c.heading_path}\n{c.text}" for c in chunks])

        for case in [c for c in cases if c.character == character]:
            query_vec = encode([f"{query_prefix}{case.query}"])[0]
            sims = vectors @ query_vec

            best = int(np.argmax(sims))
            if case.is_unrelated:
                # 답이 없는 질의가 끌어오는 최고 유사도 — 임계값의 하한을 정한다
                similarity = float(sims[best])
                correct = False
            else:
                hits = [i for i, ch in enumerate(chunks) if _matches(ch, case.expected)]
                similarity = float(max(sims[i] for i in hits)) if hits else float("nan")
                correct = best in hits

            outcomes.append(
                Outcome(
                    case=case,
                    similarity=similarity,
                    top1_heading=chunks[best].heading_path,
                    top1_similarity=float(sims[best]),
                    top1_correct=correct,
                )
            )

    return outcomes


def summarize(outcomes: list[Outcome]) -> dict:
    related = [o for o in outcomes if not o.case.is_unrelated]
    noise = [o for o in outcomes if o.case.is_unrelated]

    def band(name: str) -> list[Outcome]:
        return [o for o in related if o.case.band == name]

    def accuracy(items: list[Outcome]) -> float:
        return round(sum(1 for o in items if o.top1_correct) / len(items), 3) if items else 0.0

    hit_sims = [o.similarity for o in related]
    noise_sims = [o.similarity for o in noise]

    return {
        "hit_mean": round(float(np.mean(hit_sims)), 3),
        "hit_min": round(float(np.min(hit_sims)), 3),
        "noise_mean": round(float(np.mean(noise_sims)), 3),
        "noise_max": round(float(np.max(noise_sims)), 3),
        "separation": round(float(np.mean(hit_sims) - np.mean(noise_sims)), 3),
        "margin": round(float(np.min(hit_sims) - np.max(noise_sims)), 3),
        "top1_all": accuracy(related),
        "top1_paraphrase": accuracy(band("paraphrase")),
        "top1_literal": accuracy(band("literal")),
    }


def format_table(rows: list[tuple[str, dict]]) -> str:
    lines = [
        "",
        "=== 한국어 변별력 ===",
        "",
        f"{'변형':<18}{'적중평균':>9}{'적중최소':>9}{'잡음평균':>9}{'잡음최대':>9}"
        f"{'변별력':>8}{'여유':>8}{'top1':>7}{'의역':>7}{'직역':>7}",
        "-" * 100,
    ]
    for name, s in rows:
        lines.append(
            f"{name:<18}{s['hit_mean']:>9.3f}{s['hit_min']:>9.3f}"
            f"{s['noise_mean']:>9.3f}{s['noise_max']:>9.3f}"
            f"{s['separation']:>8.3f}{s['margin']:>8.3f}"
            f"{s['top1_all']:>7.0%}{s['top1_paraphrase']:>7.0%}{s['top1_literal']:>7.0%}"
        )
    lines += [
        "",
        "변별력 = 적중평균 - 잡음평균   여유 = 적중최소 - 잡음최대 (음수면 임계값으로 가를 수 없다)",
        "top1 = 임베딩 단독으로 정답 조각을 1위로 올린 비율",
    ]
    return "\n".join(lines)


def format_sweep(name: str, outcomes: list[Outcome]) -> str:
    """임계값별 통과·차단율.

    절대 임계값이 쓸 만한지 판정하는 표다. 어느 값에서도 적중 통과와 잡음
    차단이 함께 높아지지 않으면 임계값 접근 자체를 버려야 한다.
    """
    related = [o for o in outcomes if not o.case.is_unrelated]
    noise = [o for o in outcomes if o.case.is_unrelated]

    lines = [
        "",
        f"=== 임계값 스윕 · {name} ===",
        "",
        f"{'임계':<8}{'적중통과':>10}{'잡음차단':>10}",
    ]
    lines.append("-" * 30)
    for t in THRESHOLDS:
        passed = sum(1 for o in related if o.similarity >= t) / len(related)
        blocked = sum(1 for o in noise if o.similarity < t) / len(noise)
        lines.append(f"{t:<8.2f}{passed:>10.0%}{blocked:>10.0%}")
    return "\n".join(lines)


def format_misses(name: str, outcomes: list[Outcome]) -> str:
    lines = ["", f"=== {name} · top-1 실패 ===", ""]
    misses = [o for o in outcomes if not o.case.is_unrelated and not o.top1_correct]
    if not misses:
        lines.append("없음")
        return "\n".join(lines)
    for o in misses:
        lines.append(
            f"  {o.case.query[:34]:<36} 기대 {o.case.expected[0][:22]:<24} "
            f"(적중 {o.similarity:.3f})  →  1위 {o.top1_heading[:26]} ({o.top1_similarity:.3f})"
        )
    return "\n".join(lines)


# ─── 하이브리드 스윕 (REQ-21-6) ───
#
# 임베딩 단독 랭킹이 아니라 **실제 검색 점수**로 잰다. 임베딩을 어떻게 섞을지가
# 결정 대상이므로, 키워드와 합산된 뒤의 결과가 아니면 근거가 되지 않는다.

E5 = "intfloat/multilingual-e5-small"

# (표시명, 모델, 절대임계 or None, 상위 K or None, 가중치, 무키워드 게이트 or None)
#
# 무키워드 게이트: 키워드가 **하나도 걸리지 않은** 조각을 임베딩만으로 후보에
# 올릴 때 요구하는 유사도. 키워드가 걸린 조각에는 적용하지 않는다 — 이미 다른
# 근거가 있으므로 임베딩은 순위만 조정하면 된다.
STRATEGIES = [
    ("현재(MiniLM)", "all-MiniLM-L6-v2", 0.45, None, 0.3, None),
    ("키워드만", None, None, None, 0.0, None),
    ("e5 · 임계0.45", E5, 0.45, None, 0.3, None),
    ("e5 · 임계0.85", E5, 0.85, None, 0.3, None),
    ("e5 · 상위2", E5, None, 2, 0.3, None),
    ("e5 · 상위2+게이트0.82", E5, None, 2, 0.3, 0.82),
    ("e5 · 상위2+게이트0.85", E5, None, 2, 0.3, 0.85),
    ("e5 · 상위3+게이트0.85", E5, None, 3, 0.3, 0.85),
    ("e5 · 게이트0.85만", E5, None, None, 0.3, 0.85),
    ("e5 · 상위2+게이트0.85 w0.5", E5, None, 2, 0.5, 0.85),
]


def run_strategy(
    cases: list[Case],
    model_id: str | None,
    min_similarity: float | None,
    top_k: int | None,
    weight: float,
    keyword_gate: float | None = None,
) -> dict:
    """실제 검색 점수 구조를 재현해 최종 랭킹과 반환량을 잰다.

    `MIN_SEARCH_SCORE` 하한과 키워드 점수는 모듈에서 그대로 가져온다 —
    재현이 실물과 어긋나면 측정이 근거가 되지 못한다.
    """
    from src.modules.knowledge import _WORD, MIN_SEARCH_SCORE

    encode = None
    if model_id is not None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id)
        encode = lambda texts: model.encode(texts, normalize_embeddings=True)  # noqa: E731

    scorer = KnowledgeModule("")
    hit_top1 = miss_top1 = 0
    related_returned: list[int] = []
    noise_returned: list[int] = []

    for character in sorted({c.character for c in cases}):
        chunks = load_chunks(character)
        vectors = encode([f"{c.heading_path}\n{c.text}" for c in chunks]) if encode else None

        for case in [c for c in cases if c.character == character]:
            words = set(_WORD.findall(case.query.lower()))
            scores = [scorer._keyword_score(words, f"{c.heading_path}\n{c.text}") for c in chunks]

            if vectors is not None:
                sims = vectors @ encode([case.query])[0]
                allowed = set(range(len(chunks)))
                if min_similarity is not None:
                    allowed = {i for i in allowed if sims[i] >= min_similarity}
                if top_k is not None:
                    allowed = set(sorted(allowed, key=lambda i: -sims[i])[:top_k])
                if keyword_gate is not None:
                    # 키워드 근거가 없는 조각은 더 높은 유사도를 요구한다
                    allowed = {
                        i for i in allowed if scores[i] > 0 or float(sims[i]) >= keyword_gate
                    }
                for i in allowed:
                    scores[i] += float(sims[i]) * weight

            ranked = sorted(
                (i for i in range(len(chunks)) if scores[i] >= MIN_SEARCH_SCORE),
                key=lambda i: -scores[i],
            )

            if case.is_unrelated:
                noise_returned.append(len(ranked))
                continue

            related_returned.append(len(ranked))
            hits = [i for i, ch in enumerate(chunks) if _matches(ch, case.expected)]
            if ranked and ranked[0] in hits:
                hit_top1 += 1
            else:
                miss_top1 += 1

    total = hit_top1 + miss_top1
    return {
        "top1": round(hit_top1 / total, 3) if total else 0.0,
        "related_returned": round(float(np.mean(related_returned)), 1),
        "noise_returned": round(float(np.mean(noise_returned)), 1),
        "noise_silent": round(sum(1 for n in noise_returned if n == 0) / len(noise_returned), 3),
    }


def format_hybrid(rows: list[tuple[str, dict]]) -> str:
    lines = [
        "",
        "=== 하이브리드 검색 (키워드 + 임베딩) ===",
        "",
        f"{'전략':<26}{'top1':>8}{'적중시 반환':>12}{'무관시 반환':>12}{'무관 무응답':>12}",
        "-" * 72,
    ]
    for name, s in rows:
        lines.append(
            f"{name:<26}{s['top1']:>8.0%}{s['related_returned']:>12.1f}"
            f"{s['noise_returned']:>12.1f}{s['noise_silent']:>12.0%}"
        )
    lines += [
        "",
        "top1 = 최종 점수 1위가 정답 조각인 비율",
        "무관 무응답 = 자료에 답이 없는 질의에 아무것도 반환하지 않은 비율 (높을수록 좋다)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="임베딩 모델의 한국어 변별력 측정")
    parser.add_argument("--json", help="결과를 JSON으로 저장할 경로")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="키워드와 합산한 최종 검색 점수로 전략을 비교한다 (임베딩 반영 방식의 선택 근거)",
    )
    args = parser.parse_args(argv)

    cases = load_cases()

    if args.hybrid:
        rows = []
        for name, model_id, min_sim, top_k, weight, gate in STRATEGIES:
            print(f"[{name}] 측정 중...", flush=True)
            rows.append((name, run_strategy(cases, model_id, min_sim, top_k, weight, gate)))
        print(format_hybrid(rows))
        if args.json:
            Path(args.json).write_text(
                json.dumps(dict(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"\n저장: {args.json}")
        return 0
    rows: list[tuple[str, dict]] = []
    detail: dict[str, list[Outcome]] = {}

    for name, model_id, query_prefix, passage_prefix in VARIANTS:
        print(f"[{name}] 측정 중...", flush=True)
        outcomes = run_variant(cases, model_id, query_prefix, passage_prefix)
        detail[name] = outcomes
        rows.append((name, summarize(outcomes)))

    print(format_table(rows))
    for name, outcomes in detail.items():
        print(format_sweep(name, outcomes))
    for name, outcomes in detail.items():
        print(format_misses(name, outcomes))

    if args.json:
        payload = {
            "cases": len(cases),
            "variants": {
                name: {
                    "summary": summary,
                    "outcomes": [
                        {
                            "query": o.case.query,
                            "band": o.case.band,
                            "similarity": round(o.similarity, 4),
                            "top1_heading": o.top1_heading,
                            "top1_similarity": round(o.top1_similarity, 4),
                            "top1_correct": o.top1_correct,
                        }
                        for o in detail[name]
                    ],
                }
                for name, summary in rows
            },
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n저장: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
