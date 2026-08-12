"""few-shot 검색 정밀도 측정 (TASK-15 / REQ-15-2, REQ-15-4).

    PYTHONPATH=. uv run python -m eval.fewshot_probe
    PYTHONPATH=. uv run python -m eval.fewshot_probe --json out.json

LLM API 키가 필요 없다. 임베딩 모델은 로컬(`all-MiniLM-L6-v2`)에서 돈다.
기본 테스트 스위트에는 넣지 않는다 — 모델 로드에 수 초가 걸리고, 정밀도는
합격/불합격이 아니라 **추적할 수치**이기 때문이다.

무엇을 재는가:

    tag 정확도   검색된 예시가 기대한 태그 그룹에서 나왔는가
    무응답률     아무 예시도 반환하지 않은 비율

`expected_tag: null`(unrelated)인 질의는 **아무것도 반환하지 않는 것이 정답**이다.
관련 없는 예시를 프롬프트에 넣으면 응답 품질이 조용히 떨어진다.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.character_layout import CharacterLayout
from src.modules.fewshot import FewShotModule

PROBE_PATH = Path(__file__).parent / "datasets" / "fewshot_probe.yaml"
CHARACTERS_DIR = Path("characters")


@dataclass(frozen=True)
class Probe:
    character: str
    query: str
    expected_tag: str | None
    band: str


@dataclass
class Outcome:
    probe: Probe
    returned_tag: str | None
    correct: bool


def load_probes(path: Path = PROBE_PATH) -> list[Probe]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Probe(
            character=c["character"],
            query=c["query"],
            expected_tag=c.get("expected_tag"),
            band=c.get("band", "?"),
        )
        for c in payload["cases"]
    ]


def _embedding_fn():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return lambda text: model.encode(text, normalize_embeddings=True)


def _tag_of(module: FewShotModule, example) -> str | None:
    """검색된 예시가 어느 그룹에서 나왔는지 되짚는다."""
    for group in module.get_all_groups():
        if any(e.user == example.user and e.character == example.character for e in group.examples):
            return group.tag
    return None


def run(probes: list[Probe]) -> list[Outcome]:
    embed = _embedding_fn()
    modules: dict[str, FewShotModule] = {}
    outcomes: list[Outcome] = []

    for probe in probes:
        if probe.character not in modules:
            layout = CharacterLayout.of(CHARACTERS_DIR / probe.character)
            module = FewShotModule(str(layout.examples_dir), embedding_fn=embed)
            module.load_all()
            modules[probe.character] = module

        module = modules[probe.character]
        hits = module.search(probe.query, top_k=1)
        returned = _tag_of(module, hits[0]) if hits else None
        outcomes.append(
            Outcome(probe=probe, returned_tag=returned, correct=returned == probe.expected_tag)
        )

    return outcomes


def summarize(outcomes: list[Outcome]) -> dict:
    by_band: dict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        by_band[o.probe.band].append(o)

    def rate(items: list[Outcome]) -> float:
        return round(sum(1 for o in items if o.correct) / len(items), 3) if items else 0.0

    return {
        "total": len(outcomes),
        "accuracy": rate(outcomes),
        "by_band": {
            band: {"count": len(v), "accuracy": rate(v)} for band, v in sorted(by_band.items())
        },
        "empty_returns": sum(1 for o in outcomes if o.returned_tag is None),
    }


def format_report(outcomes: list[Outcome], summary: dict) -> str:
    lines = ["", "=== few-shot 검색 정밀도 ===", ""]
    lines.append(f"{'':<14}{'질의':<28}{'기대':<8}{'반환':<8}")
    lines.append("-" * 62)
    for o in outcomes:
        mark = "O" if o.correct else "X"
        lines.append(
            f"{mark} {o.probe.band:<12}{o.probe.query[:26]:<28}"
            f"{str(o.probe.expected_tag or '-'):<8}{str(o.returned_tag or '-'):<8}"
        )

    lines += ["", f"전체 정확도  {summary['accuracy']:.1%}  ({summary['total']}건)", ""]
    for band, stat in summary["by_band"].items():
        lines.append(f"  {band:<12}{stat['accuracy']:.1%}  ({stat['count']}건)")
    lines.append(f"\n무응답 {summary['empty_returns']}건")
    return "\n".join(lines)


def sweep(probes: list[Probe], thresholds: list[float]) -> str:
    """임계값을 바꿔가며 정확도를 잰다 — 값을 눈대중으로 고르지 않기 위함이다."""
    from src.modules import fewshot as fewshot_module

    original = fewshot_module.MIN_FEWSHOT_SCORE
    lines = ["", "=== 임계값 스윕 ===", ""]
    lines.append(
        f"{'임계값':<10}{'전체':<10}{'in-vocab':<12}{'domain':<10}{'unrelated':<12}{'무응답'}"
    )
    lines.append("-" * 62)
    try:
        for threshold in thresholds:
            fewshot_module.MIN_FEWSHOT_SCORE = threshold
            summary = summarize(run(probes))
            bands = summary["by_band"]
            lines.append(
                f"{threshold:<10.2f}{summary['accuracy']:<10.1%}"
                f"{bands.get('in-vocab', {}).get('accuracy', 0):<12.1%}"
                f"{bands.get('domain', {}).get('accuracy', 0):<10.1%}"
                f"{bands.get('unrelated', {}).get('accuracy', 0):<12.1%}"
                f"{summary['empty_returns']}"
            )
    finally:
        fewshot_module.MIN_FEWSHOT_SCORE = original

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="few-shot 검색 정밀도 측정")
    parser.add_argument("--json", help="결과를 JSON으로 저장할 경로")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="임계값을 바꿔가며 정확도를 비교한다 (MIN_FEWSHOT_SCORE 선택 근거)",
    )
    args = parser.parse_args(argv)

    probes = load_probes()

    if args.sweep:
        print(sweep(probes, [0.0, 0.20, 0.25, 0.29, 0.32, 0.35, 0.40, 0.45]))
        return 0

    outcomes = run(probes)
    summary = summarize(outcomes)
    print(format_report(outcomes, summary))

    if args.json:
        payload = {
            "summary": summary,
            "cases": [
                {
                    "character": o.probe.character,
                    "query": o.probe.query,
                    "band": o.probe.band,
                    "expected_tag": o.probe.expected_tag,
                    "returned_tag": o.returned_tag,
                    "correct": o.correct,
                }
                for o in outcomes
            ],
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n저장: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
