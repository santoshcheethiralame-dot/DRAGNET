"""Turn run artifacts into verdicts on the preregistered hypotheses — CPU only, no model.

Reads the natural cell (mscs.jsonl, roles.jsonl, predictions.jsonl) for H1, H2, and H5, and the
AND cell (validation.jsonl, extraction.jsonl, roles.jsonl, scenarios.jsonl) for H3 and H4. Any
hypothesis whose artifact is missing prints as pending instead of failing the rest — cells run
before the per-case artifacts existed cover H3a and H4 only through their run logs, until the
deterministic re-run materializes the rows.

    python scripts/score_prereg.py --natural-cell runs/hotpotqa/natural/qwen \\
        --and-cell runs/hotpotqa/and/qwen
"""
import argparse
import json
from pathlib import Path

from lineup.data.serialization import read_roles, read_scenarios

from _cells import load_cases
from scope.conformal import depth_item
from scope.prereg import (
    h1_small_sets,
    h2_coverage,
    h3_designed_sufficient,
    h3_link_silent,
    h4_paired_arms,
    h5_loo_intersection,
)


def _read_rows(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _missing(cell: Path, *names: str) -> list[str]:
    return [name for name in names if not (cell / name).exists()]


def _fmt(value, places=2):
    return f"{value:.{places}f}" if value is not None else "n/a"


def _interval(pair) -> str:
    low, high = pair
    return f" [{low:.2f}, {high:.2f}]" if low is not None else ""


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--natural-cell", type=Path, required=True)
    parser.add_argument("--and-cell", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    mscs_rows = _read_rows(args.natural_cell / "mscs.jsonl")
    if mscs_rows is None:
        print("H1  pending — no mscs.jsonl in the natural cell")
        print("H2  pending — same")
        print("H5  pending — same")
    else:
        h1 = h1_small_sets(mscs_rows, n_boot=args.bootstrap)
        print(
            f"H1  {_verdict(h1['passed'])}  small-set rate {_fmt(h1['rate'])}{_interval(h1['interval'])} "
            f">= 0.50  (n={h1['n']}, parametric fraction {_fmt(h1['parametric_fraction'])})"
        )

        needed = _missing(args.natural_cell, "scenarios.jsonl", "generations.jsonl", "predictions.jsonl")
        if needed:
            print(f"H2  pending — natural cell missing {', '.join(needed)}")
        else:
            cases = load_cases(args.natural_cell, "mscs")
            items = [depth_item(c.key, c.ranking, c.family) for c in cases if c.ranking is not None]
            for split in h2_coverage(items, alpha=args.alpha, seeds=tuple(args.seeds)):
                if "coverage" not in split:
                    print(f"H2  seed={split['seed']}  no data")
                    continue
                tau = "inf" if split["tau"] == float("inf") else f"{split['tau']:.0f}"
                print(
                    f"H2  {_verdict(split['passed'])}  seed={split['seed']}  coverage {split['coverage']:.2f} "
                    f"(target {split['target']:.2f} - 2SE {split['slack']:.2f})  "
                    f"mean-size {split['mean_size']:.1f} <= 4  tau={tau}  n_test={split['n_test']}"
                )

        if _missing(args.natural_cell, "roles.jsonl"):
            print("H5  pending — no roles.jsonl in the natural cell")
        else:
            h5 = h5_loo_intersection(mscs_rows, read_roles(args.natural_cell / "roles.jsonl"))
            print(
                f"H5  {_verdict(h5['passed'])}  loo = intersection rate {_fmt(h5['rate'])} >= 0.90  "
                f"(n={h5['n']}, violations excluded {h5['violating']}, no set within bound {h5['unevaluable']})"
            )

    validation_rows = _read_rows(args.and_cell / "validation.jsonl")
    if validation_rows is None:
        print("H3a pending — no validation.jsonl in the AND cell (rates live in the run log)")
    else:
        h3a = h3_designed_sufficient(validation_rows, n_boot=args.bootstrap)
        print(
            f"H3a {_verdict(h3a['passed'])}  designed-sufficient {_fmt(h3a['rate'])}{_interval(h3a['interval'])} "
            f">= 0.50  (n={h3a['n']})"
        )

    needed = _missing(args.and_cell, "roles.jsonl", "scenarios.jsonl")
    if needed:
        print(f"H3b pending — AND cell missing {', '.join(needed)}")
    else:
        scenarios = {s.qid: s for s in read_scenarios(args.and_cell / "scenarios.jsonl")}
        h3b = h3_link_silent(read_roles(args.and_cell / "roles.jsonl"), scenarios)
        print(
            f"H3b {_verdict(h3b['passed'])}  link silent {_fmt(h3b['link_rate'])} (n={h3b['link_n']}) "
            f"vs distractor {_fmt(h3b['distractor_rate'])} (n={h3b['distractor_n']})  >= 2x"
        )

    extraction_rows = _read_rows(args.and_cell / "extraction.jsonl")
    if extraction_rows is None:
        print("H4  pending — no extraction.jsonl in the AND cell (re-run extraction to materialize rows)")
    else:
        h4 = h4_paired_arms(extraction_rows, n_boot=max(args.bootstrap, 2000))
        print(
            f"H4  {_verdict(h4['passed'])}  beam {_fmt(h4.get('rate_a'))} vs contextcite {_fmt(h4.get('rate_b'))}  "
            f"mean diff {_fmt(h4.get('mean_diff'))}  one-sided p {_fmt(h4.get('p'), 3)} < 0.05  (n={h4['n']})"
        )


if __name__ == "__main__":
    main()
