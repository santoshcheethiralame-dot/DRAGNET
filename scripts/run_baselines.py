"""Named-baseline head-to-head over existing cells — CPU only, no model, no new runs.

Every arm is a ranking of the six passages; the families and the oracle already exist on disk, so
this reuses the exact conformal and single-passage scoring the paper reports and only varies the
ranking. Arms: presented order (the blind floor, position only), leave-one-out (causal passages
first, from roles.jsonl), ContextCite (score order, from predictions.jsonl), and DRAGNET's own
interaction and Shapley orders (from orders.jsonl). Reports, per arm: single-passage accuracy (the
top-ranked passage is itself a sufficient set), and the set-valued guarantee at the target level
(calibrated depth, coverage, mean size). This is the numerical version of the positioning table.

    python scripts/run_baselines.py --cells runs/hotpotqa/natural/qwen ... --alpha 0.1
"""
import argparse
import json
import random
from pathlib import Path

from lineup.data.serialization import read_roles, read_scenarios

from _cells import load_cases
from dragnet.conformal import calibrate_depth, coverage_and_size, depth_item
from dragnet.designed import set_covers


def presented_order(scenario) -> list[str]:
    return [c.chunk_id for c in scenario.chunks]


def loo_order(scenario, causal: set[str]) -> list[str]:
    order = presented_order(scenario)
    return [c for c in order if c in causal] + [c for c in order if c not in causal]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    # collect, per arm, the (case-key, ranking, family) items and the top-1 single-passage hits
    arms = ["presented", "loo", "contextcite", "interaction", "shapley"]
    items: dict[str, list] = {a: [] for a in arms}
    top1: dict[str, list[int]] = {a: [] for a in arms}

    for cell in args.cells:
        scenarios = {s.qid: s for s in read_scenarios(cell / "scenarios.jsonl")}
        causal = {}
        if (cell / "roles.jsonl").exists():
            for rec in read_roles(cell / "roles.jsonl"):
                causal[rec.qid] = {r.chunk_id for r in rec.chunk_roles if getattr(r, "causal", False)}
        for case in load_cases(cell, "mscs"):
            if not case.family:
                continue
            scenario = scenarios.get(case.key.split("/")[-1]) or scenarios.get(case.key)
            if scenario is None:
                continue
            rankings = {
                "presented": presented_order(scenario),
                "loo": loo_order(scenario, causal.get(scenario.qid, set())),
                "contextcite": case.ranking,
                "interaction": case.interaction,
                "shapley": case.shapley,
            }
            for arm, ranking in rankings.items():
                if not ranking:
                    continue
                items[arm].append(depth_item(case.key, ranking, case.family))
                top1[arm].append(int(set_covers(frozenset({ranking[0]}), case.family)))

    print(f"== named-baseline head-to-head (alpha={args.alpha}, pooled over {len(args.cells)} cells) ==")
    print(f"{'arm':13} {'n':>4}  {'single-passage acc':>18}  {'tau':>4}  {'coverage':>9}  {'mean size':>9}")
    print("-" * 68)
    for arm in arms:
        if not items[arm]:
            print(f"{arm:13} {'--':>4}  (no ranking on disk)")
            continue
        n = len(items[arm])
        s1 = sum(top1[arm]) / n
        taus, covs, sizes = [], [], []
        for seed in args.seeds:
            shuffled = list(items[arm])
            random.Random(seed).shuffle(shuffled)
            half = len(shuffled) // 2
            tau = calibrate_depth(shuffled[:half], alpha=args.alpha)
            covered, size = coverage_and_size(shuffled[half:], tau)
            taus.append(tau); covs.append(covered); sizes.append(size)
        tau_label = "/".join(sorted({("inf" if t == float("inf") else f"{t:.0f}") for t in taus}))
        cov = sum(covs) / len(covs)
        size = sum(sizes) / len(sizes)
        print(f"{arm:13} {n:>4}  {s1:>18.2f}  {tau_label:>4}  {cov:>9.2f}  {size:>9.1f}")

    print("\nsingle-passage acc = the top-ranked passage alone reproduces the answer (a singleton "
          "sufficient set); coverage/size are the set-valued guarantee at the calibrated depth.")


if __name__ == "__main__":
    main()
