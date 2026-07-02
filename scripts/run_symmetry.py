"""Measure the approximate-symmetry rate on run cells — CPU only, no model.

For every wrong case, find near-duplicate passage pairs (token Jaccard) and report how often a
case carries one at all and how often one touches the responsible candidates — the empirical
frequency of the regime where Theorem 5 says a single responsible set is non-identifiable.
Members come from roles.jsonl (causal or salient passages); when the cell has an mscs.jsonl,
the enumerated minimal-sufficient union is reported as a second, sharper member set. The full
threshold curve is printed so no operating point is chosen after seeing the data.

    python scripts/run_symmetry.py --cells path/to/natural_data/hotpotqa/qwen path/to/...
"""
import argparse
import json
from pathlib import Path

from lineup.data.serialization import read_roles, read_scenarios

from scope.symmetry import case_symmetry, symmetry_rates

THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _mscs_records(cell: Path) -> dict:
    path = cell / "mscs.jsonl"
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[record["qid"]] = [frozenset(subset) for subset in record["minimal_sufficient"]]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    cases = []
    families = []
    for cell in args.cells:
        scenarios = {s.qid: s for s in read_scenarios(cell / "scenarios.jsonl")}
        mscs = _mscs_records(cell)
        for case in read_roles(cell / "roles.jsonl"):
            if case.original_correct:
                continue
            scenario = scenarios.get(case.qid)
            if scenario is None:
                continue
            role_members = frozenset(r.chunk_id for r in case.chunk_roles if r.causal or r.salient)
            family = mscs.get(case.qid)
            members = frozenset(x for subset in family for x in subset) if family else None
            cases.append((str(cell), scenario.chunks, role_members, members))
            if family is not None:
                families.append(family)

    print(f"cells={len(args.cells)}  wrong cases={len(cases)}")
    print("\nthreshold curve (pooled; members = causal-or-salient roles):")
    for threshold in THRESHOLDS:
        reads = [case_symmetry(chunks, members, threshold) for _, chunks, members, _ in cases]
        rates = symmetry_rates(reads)
        print(
            f"  t={threshold:.1f}  near-dup case rate {rates['pair_rate']:.2f}  "
            f"touching responsible candidates {rates['member_rate']:.2f}"
        )

    if families:
        print("\nthreshold curve (members = enumerated minimal-sufficient union):")
        eligible = [(chunks, mscs) for _, chunks, _, mscs in cases if mscs is not None]
        for threshold in THRESHOLDS:
            reads = [case_symmetry(chunks, members, threshold) for chunks, members in eligible]
            rates = symmetry_rates(reads)
            print(
                f"  t={threshold:.1f}  n={rates['n']}  near-dup case rate {rates['pair_rate']:.2f}  "
                f"touching a member {rates['member_rate']:.2f}"
            )
        # Functional interchangeability needs no text at all: more than one minimal sufficient
        # set IS the symmetry precondition, whatever the passages look like.
        nonparam = [family for family in families if frozenset() not in family]
        ambiguous = sum(1 for family in nonparam if len(family) > 1)
        if nonparam:
            print(
                f"\nfunctional ambiguity (>1 minimal sufficient set, non-parametric): "
                f"{ambiguous}/{len(nonparam)} = {ambiguous / len(nonparam):.2f}"
            )

    closest = sorted(
        (case_symmetry(chunks, members, 1.1)["closest"] for _, chunks, members, _ in cases),
        reverse=True,
    )
    if closest:
        n = len(closest)
        print("\nclosest-pair similarity per case (the observable gap):")
        print(f"  max {closest[0]:.2f}  p90 {closest[n // 10]:.2f}  median {closest[n // 2]:.2f}")

    print("\nper cell (t=0.6):")
    for cell in args.cells:
        subset = [(chunks, members) for source, chunks, members, _ in cases if source == str(cell)]
        reads = [case_symmetry(chunks, members, 0.6) for chunks, members in subset]
        rates = symmetry_rates(reads)
        if rates["n"]:
            print(
                f"  {cell}  n={rates['n']}  pair rate {rates['pair_rate']:.2f}  "
                f"touching {rates['member_rate']:.2f}"
            )


if __name__ == "__main__":
    main()
