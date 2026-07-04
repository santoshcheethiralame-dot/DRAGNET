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

    # 2 — disjoint explanations, with case-bootstrap intervals
    flags = []   # (ambiguous, disjoint) per evaluable case
    multiplicity: Counter = Counter()
    for d in data.values():
        for family in d["families"].values():
            if not family:
                continue
            multiplicity[min(len(family), 5)] += 1
            amb = len(family) > 1
            flags.append((amb, amb and any(not (a & b) for a, b in combinations(family, 2))))
    evaluable = len(flags)
    ambiguous = sum(a for a, _ in flags)
    disjoint = sum(d for _, d in flags)

    def ci(select, n_boot=1000, seed=0):
        rng = random.Random(seed)
        draws = sorted(
            sum(select(flags[rng.randrange(evaluable)]) for _ in range(evaluable)) / evaluable
            for _ in range(n_boot)
        )
        return draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]

    amb_lo, amb_hi = ci(lambda f: f[0])
    dis_lo, dis_hi = ci(lambda f: f[1])
    print(f"\n== disjointness (n evaluable={evaluable}) ==")
    print(f"  ambiguous (>1 member): {ambiguous / evaluable:.2f} [{amb_lo:.2f}, {amb_hi:.2f}]")
    print(f"  fully disjoint: {disjoint / evaluable:.2f} [{dis_lo:.2f}, {dis_hi:.2f}] of evaluable "
          f"({disjoint}/{ambiguous} = {disjoint / ambiguous:.2f} of ambiguous)")
    print(f"  member-count histogram (cap 5): {dict(sorted(multiplicity.items()))}")

    # 2b — heterogeneity by question type (meta 'type', else the musique hop prefix)
    by_type: dict[str, list] = {}
    for cell, d in data.items():
        kinds = {}
        for s in read_scenarios(cell / "scenarios.jsonl"):
            kind = (s.meta or {}).get("type") or (s.qid.split("__")[0] if "__" in s.qid else None)
            kinds[s.qid] = kind or "unknown"
        for qid, family in d["families"].items():
            if family:
                amb = len(family) > 1
                dis = amb and any(not (a & b) for a, b in combinations(family, 2))
                by_type.setdefault(kinds.get(qid, "unknown"), []).append((amb, dis))
    print("\n== heterogeneity by question type (groups with n>=20) ==")
    for kind, fl in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        if len(fl) < 20:
            continue
        n = len(fl)
        print(f"  {kind:12s} n={n:4d}  ambiguity {sum(a for a, _ in fl) / n:.2f}  "
              f"disjoint {sum(d for _, d in fl) / n:.2f}")

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

    # 5b — member vs non-member passage length (the companion confound check)
    member_len, other_len = [], []
    for cell, d in data.items():
        texts = {c.chunk_id: len(c.text) for s in read_scenarios(cell / "scenarios.jsonl") for c in s.chunks}
        seen_by_qid = {s.qid: [c.chunk_id for c in s.chunks] for s in read_scenarios(cell / "scenarios.jsonl")}
        for qid, family in d["families"].items():
            if not family:
                continue
            members = frozenset().union(*family)
            for cid in seen_by_qid.get(qid, []):
                (member_len if cid in members else other_len).append(texts[cid])
    if member_len and other_len:
        print(f"\n== member vs non-member passage length ==")
        print(f"  members mean {sum(member_len) / len(member_len):.0f} chars   "
              f"non-members {sum(other_len) / len(other_len):.0f} chars")

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
