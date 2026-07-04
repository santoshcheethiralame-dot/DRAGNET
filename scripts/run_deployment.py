"""Deployment-grade checks — the query-budget frontier and the transfer of the guarantee.

Budget frontier: extraction.jsonl logs the queries each arm spent per case; joined with the
behavioral families from validation.jsonl (where present), the frontier is coverage per query
budget — the honest differentiator when every arm reaches sufficiency.

Tau-transfer: calibrate the conformal depth on one dataset's cells and evaluate coverage on
another's — does the guarantee travel, or is it a per-benchmark artifact? Run within one model
(the exchangeability-honest version).

    python scripts/run_deployment.py --natural-cells <cells...> --and-cells <cells...> --alpha 0.1
"""
import argparse
import json
from pathlib import Path

from _cells import load_cases
from dragnet.conformal import calibrate_depth, coverage_and_size, depth_item


def rows_of(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--natural-cells", type=Path, nargs="+", required=True)
    parser.add_argument("--and-cells", type=Path, nargs="*", default=[])
    parser.add_argument("--arm", default="shapley", help="ranking arm for the transfer test")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    if args.and_cells:
        print("== query-budget frontier (per arm: mean queries vs behavioral coverage) ==")
        for cell in args.and_cells:
            extraction = rows_of(cell / "extraction.jsonl")
            if extraction is None:
                continue
            validation = rows_of(cell / "validation.jsonl")
            families = {}
            if validation:
                families = {
                    r["qid"]: [frozenset(s) for s in r["minimal_sufficient"] if s] for r in validation
                }
            arms: dict[str, dict] = {}
            for r in extraction:
                a = arms.setdefault(r["arm"], {"n": 0, "queries": 0, "covered": 0, "cov_n": 0, "suff": 0})
                a["n"] += 1
                a["queries"] += r["queries"]
                a["suff"] += r["sufficient"]
                fam = families.get(r["qid"])
                if fam:
                    a["cov_n"] += 1
                    a["covered"] += any(m <= frozenset(r["subset"]) for m in fam)
            print(f"  {'/'.join(cell.parts[-3:])}:")
            for arm, a in sorted(arms.items(), key=lambda kv: kv[1]["queries"] / kv[1]["n"]):
                cov = f"behavioral-cov {a['covered'] / a['cov_n']:.2f}" if a["cov_n"] else "cov n/a"
                print(f"    {arm:12s} mean queries {a['queries'] / a['n']:5.1f}   "
                      f"sufficient {a['suff'] / a['n']:.2f}   {cov}")

    print(f"\n== tau-transfer ({args.arm} order, alpha={args.alpha}): calibrate on A, test on B ==")
    per_dataset: dict[str, list] = {}
    for cell in args.natural_cells:
        cases = load_cases(cell, "mscs")
        items = [depth_item(c.key, getattr(c, args.arm), c.family) for c in cases if getattr(c, args.arm)]
        if items:
            per_dataset.setdefault(cell.parts[-3], []).extend(items)
    names = sorted(per_dataset)
    for source in names:
        tau = calibrate_depth(per_dataset[source], alpha=args.alpha)
        label = "inf" if tau == float("inf") else f"{tau:.0f}"
        for target in names:
            if target == source:
                continue
            covered, size = coverage_and_size(per_dataset[target], tau)
            print(f"  {source:9s} (tau={label}) -> {target:9s}:  coverage {covered:.2f}  mean size {size:.1f}  "
                  f"(n={len(per_dataset[target])})")


if __name__ == "__main__":
    main()
