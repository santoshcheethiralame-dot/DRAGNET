"""The unified decision rule on existing run cells — CPU only, no model.

Bins the wrong cases by ContextCite's top1-top2 margin (the benchmark's abstention signal),
calibrates a conformal depth per bin on a seeded half split, and reads the verdicts off the
depths: tau = 1 names a singleton, tau <= cap returns the set, beyond that the bin abstains.
Reports the verdict shares, coverage among answered cases against the target, the context cost,
and a cap sweep — the risk-coverage table for the decision rule.

    python scripts/run_decision_rule.py --alpha 0.1 --bins 4 --cap 3 \\
        --cells path/to/results_data/hotpotqa/hardtraps/qwen path/to/...
"""
import argparse
import dataclasses
import random
from pathlib import Path

from _cells import load_cases
from scope.conformal import depth_item
from scope.decide import calibrate_rule, calibrate_stratified_rules, evaluate_rule, evaluate_stratified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--cap", type=int, default=3)
    parser.add_argument("--family", choices=("designed", "behavioral", "fixer"), default="behavioral")
    parser.add_argument(
        "--stratify", choices=("none", "model", "condition", "dataset"), default="none",
        help="calibrate one rule per stratum so the guarantee holds conditionally on it",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    part = {"model": -1, "condition": -2, "dataset": -3}
    by_stratum: dict = {}
    for cell in args.cells:
        for case in load_cases(cell, args.family):
            if case.ranking is None:
                continue
            stratum = Path(case.source).parts[part[args.stratify]] if args.stratify != "none" else "all"
            by_stratum.setdefault(stratum, []).append(
                (case.margin, depth_item(case.key, case.ranking, case.family))
            )

    # Split within each stratum, so no stratum ends up calibration-starved by the shuffle.
    calibration_by, test_by = {}, {}
    for stratum, stratum_pairs in sorted(by_stratum.items()):
        rng = random.Random(args.seed)
        rng.shuffle(stratum_pairs)
        half = len(stratum_pairs) // 2
        calibration_by[stratum], test_by[stratum] = stratum_pairs[:half], stratum_pairs[half:]

    if args.stratify != "none":
        rules = calibrate_stratified_rules(calibration_by, alpha=args.alpha, bins=args.bins, cap=args.cap)
        stratified = evaluate_stratified(rules, test_by)
        print(f"stratify={args.stratify}  alpha={args.alpha}  bins={args.bins}  cap={args.cap}")
        for stratum, row in sorted(stratified["per_stratum"].items()):
            coverage = f"{row['answered_coverage']:.2f}" if row["answered_coverage"] is not None else "  --"
            size = f"{row['mean_answered_size']:.1f}" if row["mean_answered_size"] is not None else " --"
            print(
                f"  {stratum:10s} n={row['n']:4d}  abstain {row['abstain_rate']:.2f}  "
                f"coverage {coverage}  mean-size {size}"
            )
        pooled = stratified["pooled"]
        print(
            f"pooled: n={pooled['n']}  abstain {pooled['abstain_rate']:.2f}  "
            f"answered-coverage {pooled['answered_coverage']:.2f} (target {1 - args.alpha:.2f})"
        )
        return

    calibration = [pair for pairs in calibration_by.values() for pair in pairs]
    test = [pair for pairs in test_by.values() for pair in pairs]
    rule = calibrate_rule(calibration, alpha=args.alpha, bins=args.bins, cap=args.cap)
    report = evaluate_rule(rule, test)

    print(f"family={args.family}  n_cal={len(calibration)}  n_test={len(test)}  alpha={args.alpha}  cap={args.cap}")
    lows = ("-inf",) + tuple(f"{edge:.3f}" for edge in rule.edges)
    highs = tuple(f"{edge:.3f}" for edge in rule.edges) + ("inf",)
    for index, bucket in enumerate(report["bins"]):
        tau = rule.taus[index]
        verdict = "abstain" if tau is None or tau == float("inf") or (args.cap and tau > args.cap) else (
            "singleton" if tau == 1 else f"set({tau:.0f})"
        )
        coverage = f"{bucket['covered'] / bucket['answered']:.2f}" if bucket["answered"] else "  --"
        print(
            f"  bin{index} margin ({lows[index]}, {highs[index]}]  n={bucket['n']:4d}  "
            f"tau={'inf' if tau in (None, float('inf')) else f'{tau:.0f}'}  verdict={verdict:10s}  coverage {coverage}"
        )
    coverage = f"{report['answered_coverage']:.2f}" if report["answered_coverage"] is not None else "--"
    size = f"{report['mean_answered_size']:.1f}" if report["mean_answered_size"] is not None else "--"
    print(
        f"verdicts: abstain {report['abstain_rate']:.2f}  singleton {report['singleton_rate']:.2f}  "
        f"answered-coverage {coverage} (target {1 - args.alpha:.2f})  mean-size {size}"
    )

    print("\ncap sweep (risk-coverage):")
    for cap in (1, 2, 3, 4, None):
        swept = dataclasses.replace(rule, cap=cap)
        row = evaluate_rule(swept, test)
        coverage = f"{row['answered_coverage']:.2f}" if row["answered_coverage"] is not None else "--"
        size = f"{row['mean_answered_size']:.1f}" if row["mean_answered_size"] is not None else "--"
        print(
            f"  cap={str(cap):4s} answered {1 - row['abstain_rate']:.2f}  coverage {coverage}  mean-size {size}"
        )


if __name__ == "__main__":
    main()
