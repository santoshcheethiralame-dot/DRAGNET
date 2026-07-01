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
from scope.decide import calibrate_rule, evaluate_rule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--cap", type=int, default=3)
    parser.add_argument("--family", choices=("designed", "behavioral"), default="behavioral")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs = []
    for cell in args.cells:
        for case in load_cases(cell, args.family):
            if case.ranking is None:
                continue
            pairs.append((case.margin, depth_item(case.key, case.ranking, case.family)))

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    half = len(pairs) // 2
    calibration, test = pairs[:half], pairs[half:]

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
    print(
        f"verdicts: abstain {report['abstain_rate']:.2f}  singleton {report['singleton_rate']:.2f}  "
        f"answered-coverage {report['answered_coverage']:.2f} (target {1 - args.alpha:.2f})  "
        f"mean-size {report['mean_answered_size']:.1f}"
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
