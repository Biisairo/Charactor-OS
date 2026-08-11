"""평가 실행 CLI.

    uv run python -m eval.run --character hong-gil-dong
    uv run python -m eval.run --character hong-gil-dong --compare
    uv run python -m eval.run --character hong-gil-dong --dry-run

--dry-run은 LLM을 전혀 호출하지 않고 실행 경로만 점검한다 (비용 0).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from eval.config import EvalConfigError, load_judge_config, target_model_name
from eval.dataset import DatasetError, load_dataset
from eval.judge import Judge, StubJudge
from eval.runner import RESULTS_DIR, EvalRun, latency_stats, run_evaluation
from eval.scoring import (
    average_across_runs,
    format_comparison,
    format_summary,
    paired_compare,
    run_spread,
)

CHARACTERS_DIR = Path("characters")

SETTING_REVIEW_ON = "reflection-on"
SETTING_REVIEW_OFF = "reflection-off"


class _StubClient:
    """--dry-run 전용. 대화 LLM 호출을 대체한다."""

    def __init__(self) -> None:
        self.env = type("Env", (), {"model": "stub"})()

    def call_llm(self, messages, tools=None, use_stream=False, mute=True, **kwargs):
        from src.llm.client import TrimmedMessage

        last = messages[-1].get("content", "") if messages else ""
        return TrimmedMessage(
            content=f"[dry-run 응답] {str(last)[:40]}",
            role="assistant",
            reasoning_content="",
            tool_calls=[],
            usage=None,
        )


def _progress(index: int, total: int, result) -> None:
    if result.score is None:
        mark = f"실패 ({result.error})"
    else:
        mark = " ".join(f"{k}={v}" for k, v in result.score.scores.items())
    print(f"  [{index}/{total}] {result.case.id:<14} {mark}", flush=True)


def _run_one(
    args, dataset, character_dir, workspace, judge, judge_model, *, no_review, label_suffix=""
) -> EvalRun:
    setting = SETTING_REVIEW_OFF if no_review else SETTING_REVIEW_ON
    count = len(dataset.cases[: args.limit] if args.limit else dataset.cases)
    print(f"\n[{setting}]{label_suffix} 평가 시작 — 사례 {count}건")

    run = run_evaluation(
        dataset,
        character_dir,
        workspace,
        judge,
        setting=setting,
        no_review=no_review,
        target_model="stub" if args.dry_run else target_model_name(),
        judge_model=judge_model,
        client=_StubClient() if args.dry_run else None,
        limit=args.limit,
        progress=_progress,
    )

    print(f"\n[{setting}] 결과")
    print(format_summary(run.summary))

    lat = latency_stats(run.results)
    if lat["count"]:
        print(
            f"  응답 지연     평균 {lat['mean_ms'] / 1000:.1f}s"
            f" / 중앙 {lat['median_ms'] / 1000:.1f}s"
            f" / 최대 {lat['max_ms'] / 1000:.1f}s"
        )

    failed = [r for r in run.results if r.score is None]
    if failed:
        # 제외된 사례는 평균에서 빠지므로 반드시 눈에 보여야 한다.
        # 조용히 빠지면 표본이 줄어든 줄 모르고 수치를 해석하게 된다.
        print(f"\n  ⚠ 채점 제외 {len(failed)}건 (평균에서 빠짐)")
        for r in failed:
            print(f"      {r.case.id:<14} {r.error}")

    out_dir = Path(args.out) if args.out else RESULTS_DIR
    path = run.save(out_dir)
    print(f"  저장: {path}")
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Character OS 응답 품질 평가")
    parser.add_argument("--character", default="hong-gil-dong", help="평가할 캐릭터 id")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Reflection on/off 두 설정을 모두 실행하고 축별 차이를 출력",
    )
    parser.add_argument("--no-review", action="store_true", help="Reflection 비활성 설정으로 실행")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="LLM을 호출하지 않고 실행 경로만 점검 (비용 0)",
    )
    parser.add_argument("--limit", type=int, help="앞에서 N건만 평가 (디버깅용)")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="설정당 반복 실행 횟수. 2 이상이면 사례별 평균으로 판정자 변동을 줄인다",
    )
    parser.add_argument("--out", help="결과 저장 디렉토리")
    args = parser.parse_args(argv)

    # 데이터셋
    try:
        dataset = load_dataset(args.character)
    except DatasetError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2

    character_dir = CHARACTERS_DIR / args.character
    if not character_dir.exists():
        print(f"오류: 캐릭터 디렉토리가 없습니다: {character_dir}", file=sys.stderr)
        return 2

    # 판정자
    if args.dry_run:
        judge = StubJudge()
        judge_model = "stub"
        print("[dry-run] LLM을 호출하지 않습니다. 점수는 의미 없는 더미 값입니다.")
    else:
        try:
            config = load_judge_config()
        except EvalConfigError as exc:
            print(f"오류: {exc}", file=sys.stderr)
            return 2
        judge = Judge(client=config.build_client())
        judge_model = config.model
        print(f"판정 모델: {judge_model} / 대상 모델: {target_model_name()}")

    with tempfile.TemporaryDirectory(prefix="charos-eval-") as tmp:
        workspace = Path(tmp)

        if args.compare:
            repeat = max(1, args.repeat)
            runs_on: list[EvalRun] = []
            runs_off: list[EvalRun] = []

            for i in range(repeat):
                suffix = f" (반복 {i + 1}/{repeat})" if repeat > 1 else ""
                runs_on.append(
                    _run_one(
                        args,
                        dataset,
                        character_dir,
                        workspace / f"on-{i}",
                        judge,
                        judge_model,
                        no_review=False,
                        label_suffix=suffix,
                    )
                )
                runs_off.append(
                    _run_one(
                        args,
                        dataset,
                        character_dir,
                        workspace / f"off-{i}",
                        judge,
                        judge_model,
                        no_review=True,
                        label_suffix=suffix,
                    )
                )

            run_on, run_off = runs_on[-1], runs_off[-1]

            # 반복 실행했다면 사례별로 평균하여 판정자 변동을 줄인다
            scored_on = (
                average_across_runs([r.scored for r in runs_on]) if repeat > 1 else run_on.scored
            )
            scored_off = (
                average_across_runs([r.scored for r in runs_off]) if repeat > 1 else run_off.scored
            )

            if repeat > 1:
                spread_on = run_spread([r.summary for r in runs_on])
                spread_off = run_spread([r.summary for r in runs_off])
                print("\n=== 실행 간 변동 (노이즈 추정) ===")
                print(f"  on   {spread_on['values']}  변동폭 {spread_on['spread']:.2f}")
                print(f"  off  {spread_off['values']}  변동폭 {spread_off['spread']:.2f}")

            # 설정마다 채점 실패 수가 다르면 사례 집합이 어긋난다.
            # 전체 요약을 그대로 비교하면 설정 효과가 아니라 표본 차이를 보게 되므로
            # 공통으로 채점된 사례만으로 비교한다.
            pc = paired_compare(scored_off, scored_on)

            print("\n=== Reflection 효과 (공통 사례 짝 비교) ===")
            print(
                f"공통 {len(pc.common_ids)}건 "
                f"(off {len(run_off.scored)} / on {len(run_on.scored)} 채점)"
            )
            if pc.dropped_before or pc.dropped_after:
                print(
                    f"  비교 제외: {', '.join(sorted(set(pc.dropped_before) | set(pc.dropped_after)))}"
                )
            print()
            print(format_comparison("off", "on", pc.deltas))
            print(f"\n사례별: on 우세 {pc.wins_after} / off 우세 {pc.wins_before} / 동률 {pc.ties}")

            # 지연도 같은 사례 집합으로 비교한다 — Reflection의 실질 비용
            ids = set(pc.common_ids)
            lat_off = latency_stats(run_off.results, ids)
            lat_on = latency_stats(run_on.results, ids)
            if lat_off["count"] and lat_on["count"]:
                ratio = lat_on["mean_ms"] / lat_off["mean_ms"] if lat_off["mean_ms"] else 0
                print("\n=== 응답 지연 ===")
                print(f"{'':<10}{'off':>10}{'on':>10}{'배율':>10}")
                print("-" * 40)
                print(
                    f"{'평균':<10}{lat_off['mean_ms'] / 1000:>9.1f}s"
                    f"{lat_on['mean_ms'] / 1000:>9.1f}s{ratio:>9.2f}x"
                )
                print(
                    f"{'중앙':<10}{lat_off['median_ms'] / 1000:>9.1f}s"
                    f"{lat_on['median_ms'] / 1000:>9.1f}s"
                )
                print(
                    f"{'최대':<10}{lat_off['max_ms'] / 1000:>9.1f}s{lat_on['max_ms'] / 1000:>9.1f}s"
                )

            out_dir = Path(args.out) if args.out else RESULTS_DIR
            payload = pc.to_dict()
            payload["latency"] = {"off": lat_off, "on": lat_on}
            summary_path = out_dir / f"{dataset.character}_comparison_latest.json"
            summary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"\n  비교 저장: {summary_path}")
        else:
            _run_one(
                args,
                dataset,
                character_dir,
                workspace,
                judge,
                judge_model,
                no_review=args.no_review,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
