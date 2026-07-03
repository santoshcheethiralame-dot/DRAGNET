"""Structure of the enumerated families: graded responsibility and the exact spectrum — CPU only.

For every wrong case with an enumerated family, compute each passage's Chockler–Halpern degree
of responsibility and the family's exact Moebius spectrum. Prints, per cell: the distribution of
the strongest single-passage responsibility (1.0 = a true culprit exists; <= 1/2 = responsibility
is irreducibly shared — the graded form of "no single culprit"), and the fraction of cases whose
interaction spectrum violates the hierarchy assumption scalable attribution methods rely on,
with the violating pure-synergy terms counted. Parametric cases and cases with no set inside the
enumeration bound are excluded and counted.

    python scripts/run_family_structure.py --cells path/to/runs/hotpotqa/natural/qwen ...
"""
import argparse
import json
from pathlib import Path

from scope.responsibility import max_responsibility
from scope.spectrum import moebius_from_family, unsupported_terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    pooled = {"n": 0, "full": 0, "shared": 0, "minority": 0, "violating": 0, "terms": 0}
    for cell in args.cells:
        rows = [
            json.loads(line)
            for line in (cell / "mscs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        parametric = skipped = 0
        stats = {"n": 0, "full": 0, "shared": 0, "minority": 0, "violating": 0, "terms": 0}
        for row in rows:
            if row["parametric"]:
                parametric += 1
                continue
            members = [frozenset(s) for s in row["minimal_sufficient"] if s]
            if not members:
                skipped += 1
                continue
            top = max_responsibility(members)
            stats["n"] += 1
            if top == 1.0:
                stats["full"] += 1
            elif top > 0.5:
                stats["shared"] += 1
            else:
                stats["minority"] += 1
            terms = unsupported_terms(moebius_from_family(members))
            stats["violating"] += bool(terms)
            stats["terms"] += len(terms)

        n = stats["n"] or 1
        print(f"\n== {'/'.join(cell.parts[-3:])}  (n={stats['n']}, parametric {parametric}, no set in bound {skipped}) ==")
        print(
            f"  responsibility: full 1.0 on {stats['full'] / n:.2f}   "
            f"in (1/2, 1) on {stats['shared'] / n:.2f}   "
            f"<= 1/2 (irreducibly shared) on {stats['minority'] / n:.2f}"
        )
        print(
            f"  spectrum: hierarchy violated in {stats['violating'] / n:.2f} of cases "
            f"({stats['terms']} pure-synergy terms)"
        )
        for key in pooled:
            pooled[key] += stats[key]

    n = pooled["n"] or 1
    print(f"\n== pooled  (n={pooled['n']}) ==")
    print(
        f"  no passage above 1/2 responsibility: {pooled['minority'] / n:.2f}   "
        f"true culprit (1.0): {pooled['full'] / n:.2f}"
    )
    print(f"  hierarchy violated: {pooled['violating'] / n:.2f}  ({pooled['terms']} pure-synergy terms)")


if __name__ == "__main__":
    main()
