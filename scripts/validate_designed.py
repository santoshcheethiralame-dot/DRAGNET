"""Validate designed coalitions against model behavior, cell by cell.

For every wrong case, enumerate the exact minimal sufficient sets on the live model (bounded by
--max-size) and compare them to what the construction planted: how often behavior matches the
design exactly, how often the designed sets at least suffice, and how often the error turns out
parametric. Cells built before designed truth existed are retrofitted from their recipes (the
OR-covers); AND cells ship a designed.jsonl.

    python scripts/validate_designed.py --cell runs/hotpotqa/and/qwen \\
        --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --max-size 3
"""
import argparse
from pathlib import Path

from lineup.data.coalition import from_recipe, read_designed
from lineup.data.serialization import read_generations, read_scenarios

from scope.designed import compare, designed_family
from scope.model_game import scenario_game
from scope.mscs import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=Path, required=True, help="dir with scenarios/generations[/designed].jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-size", type=int, default=3, help="bound on enumerated set size")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    scenarios = {s.qid: s for s in read_scenarios(args.cell / "scenarios.jsonl")}
    wrong = [g for g in read_generations(args.cell / "generations.jsonl") if not g.is_correct]
    if args.limit:
        wrong = wrong[: args.limit]

    designed_path = args.cell / "designed.jsonl"
    if designed_path.exists():
        designed = {d.qid: d for d in read_designed(designed_path)}
    else:
        designed = {qid: d for qid, s in scenarios.items() if (d := from_recipe(s)) is not None}

    from lineup.backends import TransformersModel

    model = TransformersModel(args.model, load_in_4bit=args.load_in_4bit)

    tallies: dict = {}
    for index, generation in enumerate(wrong):
        scenario, planted = scenarios.get(generation.qid), designed.get(generation.qid)
        if scenario is None or planted is None:
            continue
        game = scenario_game(model, scenario, generation.model_answer)
        verdict = compare(
            analyze(game, max_size=args.max_size),
            designed_family(planted.cover_chunk_ids, planted.threshold),
        )
        bucket = tallies.setdefault(
            planted.structure, {"n": 0, "exact": 0, "designed_sufficient": 0, "parametric": 0, "queries": 0}
        )
        bucket["n"] += 1
        bucket["queries"] += game.queries
        for key in ("exact", "designed_sufficient", "parametric"):
            bucket[key] += int(verdict[key])
        if (index + 1) % 10 == 0:
            print(f"[{index + 1}/{len(wrong)}]", flush=True)

    for structure, t in sorted(tallies.items()):
        n = t["n"] or 1
        print(
            f"{structure:4s} n={t['n']:4d}  exact {t['exact'] / n:.2f}  "
            f"designed-sufficient {t['designed_sufficient'] / n:.2f}  "
            f"parametric {t['parametric'] / n:.2f}  mean-queries {t['queries'] / n:.0f}"
        )


if __name__ == "__main__":
    main()
