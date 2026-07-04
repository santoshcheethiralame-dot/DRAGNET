"""Interrogate the enumerated families — five analyses over cells already on disk, CPU only.

1. The oracle-optimal tau: a perfect ranking needs depth = the smallest member's size, so the
   alpha-quantile of those sizes is the floor no ranking method can beat. Reported against the
   achieved tau, per seeded half-split, pooled.
2. Disjointness: among ambiguous cases, how often two members share no passage at all — the
   maximal realization of the identifiability symmetry.
3. Cross-model overlap: scenario building is deterministic per (dataset, seed), so models saw
   identical cases; on jointly-wrong ones, do their families agree?
4. Anatomy of the unreachable: cases with no set inside the bound — by construction the full
   context is sufficient (the model reproduces its own greedy answer), so these are holistic
   errors; their A3 interference is compared against reachable cases.
5. Position: where family members sit in the presented order, against uniform.

    python scripts/run_family_analytics.py --cells <dataset>/natural/<tag> dirs... --alpha 0.1
"""
import argparse
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

from lineup.data.serialization import read_scenarios

from dragnet.conformal import DepthItem, calibrate_depth, coverage_and_size


def rows_of(cell: Path) -> list[dict]:
    return [json.loads(l) for l in (cell / "mscs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def families_of(rows: list[dict]) -> dict[str, list[frozenset[str]]]:
    return {
        r["qid"]: [frozenset(s) for s in r["minimal_sufficient"] if s]
        for r in rows
        if not r["parametric"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    data = {}
    for cell in args.cells:
        rows = rows_of(cell)
        data[cell] = {"rows": rows, "families": families_of(rows)}

    # 1 — the floor no ranking can beat
    items = []
    for cell, d in data.items():
        for qid, family in d["families"].items():
            depth = min((len(m) for m in family), default=None)
            items.append(DepthItem(key=f"{cell}/{qid}", depth=depth, n_chunks=6))
    print(f"== oracle-optimal tau (perfect ranking; pooled n={len(items)}, alpha={args.alpha}) ==")
    for seed in args.seeds:
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        half = len(shuffled) // 2
        tau = calibrate_depth(shuffled[:half], alpha=args.alpha)
        covered, size = coverage_and_size(shuffled[half:], tau)
        label = "inf" if tau == float("inf") else f"{tau:.0f}"
        print(f"  seed={seed}  tau*={label}  coverage {covered:.2f}  mean size {size:.1f}")

    # 2 — disjoint explanations
    ambiguous = disjoint = evaluable = 0
    multiplicity: Counter = Counter()
    for d in data.values():
        for family in d["families"].values():
            if not family:
                continue
            evaluable += 1
            multiplicity[min(len(family), 5)] += 1
            if len(family) > 1:
                ambiguous += 1
                if any(not (a & b) for a, b in combinations(family, 2)):
                    disjoint += 1
    print(f"\n== disjointness (n evaluable={evaluable}) ==")
    print(f"  ambiguous (>1 member): {ambiguous / evaluable:.2f}")
    print(f"  fully disjoint pair among members: {disjoint}/{ambiguous} = {disjoint / ambiguous:.2f} of ambiguous "
          f"= {disjoint / evaluable:.2f} of all evaluable")
    print(f"  member-count histogram (cap 5): {dict(sorted(multiplicity.items()))}")

    # 3 — cross-model family overlap on identical cases
    by_dataset: dict[str, dict[str, dict]] = {}
    for cell, d in data.items():
        dataset, tag = cell.parts[-3], cell.parts[-1]
        by_dataset.setdefault(dataset, {})[tag] = d["families"]
    print("\n== cross-model family overlap (same scenarios by construction) ==")
    for dataset, models in sorted(by_dataset.items()):
        for a, b in combinations(sorted(models), 2):
            common = [q for q in models[a] if q in models[b] and models[a][q] and models[b][q]]
            if not common:
                continue
            share = jacc = exact = 0.0
            for q in common:
                fa, fb = models[a][q], models[b][q]
                share += any(m in fb for m in fa)
                ua, ub = frozenset().union(*fa), frozenset().union(*fb)
                jacc += len(ua & ub) / len(ua | ub)
                exact += set(fa) == set(fb)
            n = len(common)
            print(f"  {dataset:9s} {a} vs {b}:  jointly-wrong n={n}  share-a-member {share / n:.2f}  "
                  f"union-jaccard {jacc / n:.2f}  identical-family {exact / n:.2f}")

    # 4 — the unreachable, and what destabilizes them
    print("\n== anatomy of the unreachable (no set within the bound) ==")
    for cell, d in data.items():
        rows = d["rows"]
        nonparam = [r for r in rows if not r["parametric"]]
        unreachable = [r for r in nonparam if not r["minimal_sufficient"]]
        reachable = [r for r in nonparam if r["minimal_sufficient"]]
        if not nonparam:
            continue
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        parametric = len(rows) - len(nonparam)
        print(f"  {'/'.join(cell.parts[-3:])}: unreachable {len(unreachable)}/{len(nonparam)}  "
              f"A3 violations mean {mean([r['monotonicity_violations'] for r in unreachable]):.1f} (unreachable) "
              f"vs {mean([r['monotonicity_violations'] for r in reachable]):.1f} (reachable)  "
              f"parametric {parametric}/{len(rows)}")

    # 5 — where members sit in the presented order
    position: Counter = Counter()
    baseline: Counter = Counter()
    for cell, d in data.items():
        order = {s.qid: [c.chunk_id for c in s.chunks] for s in read_scenarios(cell / "scenarios.jsonl")}
        for qid, family in d["families"].items():
            presented = order.get(qid)
            if not presented or not family:
                continue
            members = frozenset().union(*family)
            for i, cid in enumerate(presented):
                baseline[i + 1] += 1
                if cid in members:
                    position[i + 1] += 1
    print("\n== member position in the presented order (rate per slot) ==")
    print("  " + "  ".join(f"pos{i}: {position[i] / baseline[i]:.2f}" for i in sorted(baseline)))


if __name__ == "__main__":
    main()
