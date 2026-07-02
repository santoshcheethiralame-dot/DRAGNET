"""The natural-coalition slice: from a 0.10-recall single pick to a covered set. CPU only.

The slice is every wrong case with two or more individually-causal non-gold passages — organic
distributed blame, nothing planted. A single pick structurally cannot cover it (fractional
recall ~0.10 in the benchmark), and covering the *whole* responsible set is the honest target:
the depth score here is the worst rank of the set, and the calibrated prefix carries the
distribution-free guarantee that top-1 cannot even state.

Reports the slice's reproduced anchors (fractional recall@1 and oracle-sized recall@k), then
per ranking arm and alpha: the calibrated tau, test coverage of the full responsible set, and
the context cost paid.

    python scripts/run_natural_sets.py --alpha 0.1 0.2 \\
        --cells path/to/natural_data/hotpotqa/qwen path/to/natural_data/2wiki/qwen ...
"""
import argparse
import random
from pathlib import Path

from lineup.data.serialization import read_predictions, read_roles, read_scenarios

from scope.conformal import calibrate_depth, coverage_and_size, depth_item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--min-coalition", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    slice_items = {"presented": [], "contextcite": []}
    recall1_sum = recallk_sum = 0.0
    sizes = []
    total_wrong = 0
    for cell in args.cells:
        scenarios = {s.qid: s for s in read_scenarios(cell / "scenarios.jsonl")}
        rankings = {}
        for prediction in read_predictions(cell / "predictions.jsonl"):
            if prediction.method == "contextcite":
                ranked = sorted(prediction.chunk_scores, key=lambda s: s.score, reverse=True)
                rankings[prediction.qid] = [score.chunk_id for score in ranked]

        for case in read_roles(cell / "roles.jsonl"):
            if case.original_correct:
                continue
            total_wrong += 1
            scenario = scenarios.get(case.qid)
            ranking = rankings.get(case.qid)
            if scenario is None or ranking is None:
                continue
            gold = {c.chunk_id for c in scenario.chunks if c.provenance == "gold"}
            responsible = {r.chunk_id for r in case.chunk_roles if r.causal and r.chunk_id not in gold}
            if len(responsible) < args.min_coalition:
                continue

            sizes.append(len(responsible))
            recall1_sum += len(set(ranking[:1]) & responsible) / len(responsible)
            recallk_sum += len(set(ranking[: len(responsible)]) & responsible) / len(responsible)
            family = [frozenset(responsible)]     # the WHOLE set: the honest coverage target
            key = f"{cell}/{case.qid}"
            slice_items["presented"].append(depth_item(key, [c.chunk_id for c in scenario.chunks], family))
            slice_items["contextcite"].append(depth_item(key, ranking, family))

    n = len(sizes)
    if not n:
        raise SystemExit("no coalition cases found")
    print(
        f"natural-coalition slice: {n} of {total_wrong} wrong cases  mean |R| {sum(sizes) / n:.1f}  "
        f"recall@1 {recall1_sum / n:.2f}  recall@|R| {recallk_sum / n:.2f}   <- the single-pick anchors"
    )

    for arm, items in slice_items.items():
        rng = random.Random(args.seed)
        rng.shuffle(items)
        half = len(items) // 2
        calibration, test = items[:half], items[half:]
        for alpha in args.alpha:
            tau = calibrate_depth(calibration, alpha=alpha)
            covered, size = coverage_and_size(test, tau)
            tau_label = "inf" if tau == float("inf") else f"{tau:.0f}"
            print(
                f"{arm:12s} alpha={alpha:.2f}  tau={tau_label:>3s}  coverage {covered:.2f} "
                f"(target {1 - alpha:.2f})  mean-size {size:.1f}"
            )


if __name__ == "__main__":
    main()
