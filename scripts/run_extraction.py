"""Compare extraction strategies on a run cell: coverage of the designed set vs queries spent.

Five arms per wrong case, each on a fresh game so budgets are honest:
  presented    — grow-prune in the order the passages were shown (the blind floor)
  contextcite  — grow-prune ranked by ContextCite's scores from predictions.jsonl (order-1 guide)
  interaction  — sample an order-2 surrogate, grow-prune by interaction_order; the query count
                 includes the sampling masks, not just the search
  beam         — the same surrogate proposes candidate sets, verified best-first, then pruned
  shrink       — ddmin-style block removal from the full context, ContextCite order as the
                 drop priority when available (anytime: budget cuts cost minimality, not sufficiency)

Reports, per arm: sufficient-set rate, designed-set coverage (set_covers), mean set size, mean
queries. The coverage-vs-budget comparison is the seed of the efficiency figure.

    python scripts/run_extraction.py --cell runs/hotpotqa/and/qwen \\
        --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --n-samples 48
"""
import argparse
from pathlib import Path

from _cells import load_cases
from scope.designed import designed_family, set_covers
from scope.extract import grow_prune, shrink, surrogate_beam
from scope.interactions import interaction_order, sampled_effects
from scope.model_game import scenario_game

ARMS = ("presented", "contextcite", "interaction", "beam", "shrink")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--n-samples", type=int, default=48, help="surrogate masks for the interaction/beam arms")
    parser.add_argument("--budget", type=int, default=0, help="query cap per extraction; 0 = none")
    parser.add_argument("--family", choices=("designed", "behavioral"), default="designed")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cases = [case for case in load_cases(args.cell, args.family) if case.family]
    if args.limit:
        cases = cases[: args.limit]

    from lineup.backends import TransformersModel

    model = TransformersModel(args.model, load_in_4bit=args.load_in_4bit)
    budget = args.budget or None

    tallies = {arm: {"n": 0, "sufficient": 0, "covered": 0, "size": 0, "queries": 0} for arm in args.arms}
    for index, case in enumerate(cases):
        for arm in args.arms:
            if arm == "contextcite" and case.ranking is None:
                continue
            game = scenario_game(model, case.scenario, case.model_answer)
            if arm == "presented":
                result = grow_prune(game, order=list(game.ids), budget=budget)
            elif arm == "contextcite":
                result = grow_prune(game, order=case.ranking, budget=budget)
            elif arm == "interaction":
                effects = sampled_effects(game, n_samples=args.n_samples, seed=args.seed)
                result = grow_prune(game, order=interaction_order(effects), budget=budget)
            elif arm == "beam":
                effects = sampled_effects(game, n_samples=args.n_samples, seed=args.seed)
                result = surrogate_beam(game, effects, budget=budget)
            else:
                result = shrink(game, order=case.ranking or case.presented, budget=budget)

            bucket = tallies[arm]
            bucket["n"] += 1
            bucket["sufficient"] += int(result.sufficient)
            bucket["covered"] += int(set_covers(result.subset, case.family))
            bucket["size"] += len(result.subset) if result.subset else 0
            bucket["queries"] += game.queries
        if (index + 1) % 10 == 0:
            print(f"[{index + 1}/{len(cases)}]", flush=True)

    for arm in args.arms:
        t = tallies[arm]
        n = t["n"] or 1
        print(
            f"{arm:12s} n={t['n']:4d}  sufficient {t['sufficient'] / n:.2f}  covered {t['covered'] / n:.2f}  "
            f"mean-size {t['size'] / n:.1f}  mean-queries {t['queries'] / n:.0f}"
        )


if __name__ == "__main__":
    main()
