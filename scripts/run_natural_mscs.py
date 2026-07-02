"""Enumerate the behavioral minimal sufficient sets of organic errors — the real coverage target.

For every wrong case of a (typically natural) cell, walk the subset lattice up to --max-size on
the live model and record the exact minimal sufficient sets: the smallest passage subsets that
alone reproduce the wrong answer. Writes one mscs.jsonl row per case, so every downstream
analysis — conformal depths, decision rules, size distributions — re-runs on CPU without
touching the model again. Prints the aggregates that decide whether a small-set guarantee is
even on the table for organic errors: the parametric rate (the empty context already suffices),
the fraction with a small sufficient set, and the minimum-size histogram.

    python scripts/run_natural_mscs.py --cell runs/hotpotqa/natural/qwen \\
        --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --max-size 3
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from lineup.data.serialization import read_generations, read_scenarios

from scope.model_game import scenario_game
from scope.mscs import minimal_sufficient_sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-size", type=int, default=3, help="bound on enumerated set size")
    parser.add_argument("--limit", type=int, default=100, help="wrong cases to enumerate (0 = all)")
    args = parser.parse_args()

    scenarios = {s.qid: s for s in read_scenarios(args.cell / "scenarios.jsonl")}
    wrong = [g for g in read_generations(args.cell / "generations.jsonl") if not g.is_correct]
    if args.limit:
        wrong = wrong[: args.limit]

    from lineup.backends import TransformersModel

    model = TransformersModel(args.model, load_in_4bit=args.load_in_4bit)

    out = args.cell / "mscs.jsonl"
    min_sizes: Counter = Counter()
    parametric = none_found = 0
    with out.open("w", encoding="utf-8") as handle:
        for index, generation in enumerate(wrong):
            scenario = scenarios.get(generation.qid)
            if scenario is None:
                continue
            game = scenario_game(model, scenario, generation.model_answer)
            sufficient = minimal_sufficient_sets(game, max_size=args.max_size)
            record = {
                "qid": generation.qid,
                "model_answer": generation.model_answer,
                "max_size": args.max_size,
                "parametric": frozenset() in sufficient,
                "minimal_sufficient": [sorted(subset) for subset in sufficient],
                "queries": game.queries,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if record["parametric"]:
                parametric += 1
            elif sufficient:
                min_sizes[min(len(s) for s in sufficient)] += 1
            else:
                none_found += 1
            if (index + 1) % 10 == 0:
                print(f"[{index + 1}/{len(wrong)}]", flush=True)

    n = parametric + none_found + sum(min_sizes.values())
    print(f"\nwrote {out}  ({n} cases)")
    print(f"parametric (empty context suffices): {parametric}/{n}")
    print(f"no sufficient set within size {args.max_size}: {none_found}/{n}")
    print("minimum sufficient-set size:", dict(sorted(min_sizes.items())))


if __name__ == "__main__":
    main()
