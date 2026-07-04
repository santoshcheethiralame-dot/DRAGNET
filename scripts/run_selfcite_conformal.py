"""Wrap the model's own citations in the conformal layer — the trained-self-attribution bridge.

The self-citation is a parsimonious, uncalibrated proposal; the calibration layer is what turns a
proposal into a guarantee. Order each case's passages self-cited-first (cited order, then the
remaining presented order), take the prefix-depth nonconformity against the enumerated family,
and calibrate exactly as the method does. If the model's own citations earn a small tau at
target coverage, self-attribution plus calibration is a working system with no post-hoc search.

    python scripts/run_selfcite_conformal.py --cell runs/hotpotqa/natural/cerebras-gpt-oss-120b
"""
import argparse
import json
import random
from pathlib import Path

from lineup.data.serialization import read_scenarios

from dragnet.conformal import calibrate_depth, coverage_and_size, depth_item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    presented = {s.qid: [c.chunk_id for c in s.chunks] for s in read_scenarios(args.cell / "scenarios.jsonl")}
    cited = {}
    for line in (args.cell / "selfcite.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            cited[record["qid"]] = record["cited"]
    items = []
    for line in (args.cell / "mscs.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["parametric"]:
            continue
        family = tuple(frozenset(s) for s in record["minimal_sufficient"] if s)
        qid = record["qid"]
        if not family or qid not in cited or qid not in presented:
            continue
        order = list(cited[qid]) + [c for c in presented[qid] if c not in cited[qid]]
        items.append(depth_item(qid, order, family))
    print(f"cases: {len(items)}")

    for alpha in args.alphas:
        for seed in args.seeds:
            shuffled = list(items)
            random.Random(seed).shuffle(shuffled)
            half = len(shuffled) // 2
            tau = calibrate_depth(shuffled[:half], alpha=alpha)
            covered, size = coverage_and_size(shuffled[half:], tau)
            label = "inf" if tau == float("inf") else f"{tau:.0f}"
            print(f"  alpha={alpha}  seed={seed}  tau={label}  coverage {covered:.2f}  mean size {size:.1f}  "
                  f"(n_test={len(shuffled) - half})")


if __name__ == "__main__":
    main()
