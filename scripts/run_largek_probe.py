"""Deployment-scale probe: does the sufficiency structure survive k = 20-50?

The grid enumerates full families at k = 6, where the 2^k lattice is walkable. Deployment
contexts run 20-100 passages, where it is not — so this probe certifies the three headline
properties per wrong case with a bounded number of queries instead of enumerating:

    plural   — two distinct 1-minimal sufficient sets, found by two grow-prune passes; the
               second pass ranks the first set's passages last, forcing a different route
               through the context when one exists.
    disjoint — the two certified sets share no passage (the sharpest driver of the 1/m
               ceiling: no single output can even be unique).
    nonmono  — a sampled superset of a certified sufficient set fails to reproduce, the
               violation that makes leave-one-out unsound.

Every rate is a lower bound: a bounded search can miss a second route or a violating
superset, never invent one. Cost is O(k) queries per case against the lattice's 2^k —
roughly 60-80 generations at k = 30, most of them on small subsets.

The cell is built here too when it does not exist yet: the natural condition (nothing
planted) with distractors drawn by a live BM25 retriever over the corpus, so a 30-passage
context is a genuine top-30, written in the standard scenarios/generations format so any
later stage can rerun on it. An existing cell is reused, which makes an interrupted run
resumable at the probe.

    python scripts/run_largek_probe.py --cell runs/hotpotqa/largek30/qwen \\
        --dataset hotpotqa --k 30 --limit 400 \\
        --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
"""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from lineup.config import DEFAULT_MODEL, DEFAULT_SEED, set_seed
from lineup.correctness import LLMJudge
from lineup.data.retrieval import BM25DistractorRetriever, build_corpus
from lineup.data.scenario import ScenarioBuilder
from lineup.data.serialization import (
    read_generations,
    read_scenarios,
    write_generations,
    write_scenarios,
)
from lineup.data.sources import load_examples
from lineup.data.substitution import build_answer_pool
from lineup.generation import generate_and_judge

from dragnet.extract import grow_prune
from dragnet.model_game import scenario_game
from dragnet.mscs import is_sufficient


def bounded_probe(game, rng: random.Random, nonmono_samples: int, budget: int) -> dict:
    """Certify plural / disjoint / non-monotone for one case within a query budget."""
    ids = list(game.ids)

    # First pass along the presented order. The empty prefix is tested first, so the
    # parametric check comes free; an empty certified set means the model answers from
    # its own knowledge and the case leaves the evaluable pool, exactly as in the grid.
    first = grow_prune(game, order=ids, budget=budget)
    if first.subset is None:
        return {"status": "budget", "queries": game.queries}
    if not first.subset:
        return {"status": "parametric", "queries": game.queries}
    mss1 = first.subset

    # Second pass with the first set's passages ranked last: if any route to the answer
    # avoids mss1, grow reaches it before touching mss1 at all. Same set twice means the
    # bounded search found no second route — counted as not plural, the conservative side.
    order2 = [c for c in ids if c not in mss1] + [c for c in ids if c in mss1]
    second = grow_prune(game, order=order2, budget=budget)
    mss2 = second.subset
    plural = mss2 is not None and mss2 != mss1
    disjoint = plural and not (mss1 & mss2)

    # Non-monotonicity: add a few random outside passages to the certified set. One broken
    # superset is a violation; sampling this sparsely can only undercount the grid's A3.
    outside = [c for c in ids if c not in mss1]
    rng.shuffle(outside)
    nonmono = False
    tested = 0
    for extra in outside[:nonmono_samples]:
        tested += 1
        if not is_sufficient(game, mss1 | {extra}):
            nonmono = True
            break

    return {
        "status": "ok",
        "k": len(ids),
        "mss1": sorted(mss1),
        "mss2": sorted(mss2) if mss2 is not None else None,
        "plural": plural,
        "disjoint": disjoint,
        "nonmono": nonmono,
        "nonmono_tested": tested,
        "queries": game.queries,
    }


def build_cell(args, model) -> tuple[list, list]:
    examples = list(load_examples(args.dataset, args.split, limit=args.limit))
    pool = build_answer_pool(examples)
    retriever = BM25DistractorRetriever(build_corpus(examples))
    builder = ScenarioBuilder(
        answer_pool=pool, k=args.k, seed=args.seed, natural=True, retriever=retriever
    )
    judge = LLMJudge(model)

    scenarios, generations, skipped = [], [], 0
    for index, example in enumerate(examples):
        scenario = builder.build(example)
        if scenario is None:
            skipped += 1
            continue
        scenarios.append(scenario)
        generations.append(generate_and_judge(model, scenario, llm_judge=judge))
        if (index + 1) % 25 == 0:
            wrong = sum(1 for g in generations if not g.is_correct)
            print(f"[build {index + 1}/{len(examples)}] cases {len(scenarios)}  wrong {wrong}", flush=True)

    write_scenarios(args.cell / "scenarios.jsonl", scenarios)
    write_generations(args.cell / "generations.jsonl", generations)
    wrong = sum(1 for g in generations if not g.is_correct)
    print(f"built {len(scenarios)} cases at k={args.k} (skipped {skipped}), wrong {wrong}", flush=True)
    return scenarios, generations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--dataset", default="hotpotqa", help="hotpotqa, 2wiki, or musique")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=400, help="source questions to draw")
    parser.add_argument("--k", type=int, default=30, help="retrieval depth of the built contexts")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--device-map", default=None,
        help="shard an fp16 model across the visible GPUs (e.g. 'auto' for two T4s); takes "
             "precedence over --load-in-4bit. Needed at large k, where a 30-passage prefill "
             "overflows a single 16 GB card in 4-bit.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit-cases", type=int, default=120, help="wrong cases to probe (0 = all)")
    parser.add_argument("--nonmono-samples", type=int, default=4, help="sampled supersets per case")
    parser.add_argument("--budget", type=int, default=0, help="query budget per extraction (0 = 2k+8)")
    args = parser.parse_args()

    set_seed(args.seed)
    from lineup.backends import TransformersModel

    if args.device_map:
        # A single 16 GB card in 4-bit OOMs on a 30-passage prefill (weights plus the KV cache
        # and activations of a ~3-4k-token context), so shard an fp16 model across both GPUs.
        # The paper's own 4-bit-vs-fp16 check (32 vs 31% no-culprit) makes the precision change
        # immaterial to the structure this probe measures.
        model = TransformersModel(args.model, max_new_tokens=args.max_new_tokens, device_map=args.device_map)
    else:
        model = TransformersModel(args.model, max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit)
    args.cell.mkdir(parents=True, exist_ok=True)

    scenarios_path = args.cell / "scenarios.jsonl"
    generations_path = args.cell / "generations.jsonl"
    if scenarios_path.exists() and generations_path.exists():
        scenarios = list(read_scenarios(scenarios_path))
        generations = list(read_generations(generations_path))
        print(f"reusing {len(scenarios)} built cases from {args.cell}", flush=True)
    else:
        scenarios, generations = build_cell(args, model)

    by_qid = {s.qid: s for s in scenarios}
    wrong = [g for g in generations if not g.is_correct]
    if args.limit_cases:
        wrong = wrong[: args.limit_cases]
    budget = args.budget or 2 * args.k + 8
    rng = random.Random(args.seed)

    out = args.cell / "largek_probe.jsonl"
    tallies: Counter = Counter()
    sizes: list[int] = []
    queries: list[int] = []
    started = time.time()
    with out.open("w", encoding="utf-8") as handle:
        for index, generation in enumerate(wrong):
            scenario = by_qid.get(generation.qid)
            if scenario is None:
                continue
            game = scenario_game(model, scenario, generation.model_answer)
            record = bounded_probe(game, rng, args.nonmono_samples, budget)
            handle.write(json.dumps({"qid": generation.qid, "model_answer": generation.model_answer, **record}) + "\n")

            tallies[record["status"]] += 1
            queries.append(record["queries"])
            if record["status"] == "ok":
                sizes.append(len(record["mss1"]))
                for key in ("plural", "disjoint", "nonmono"):
                    tallies[key] += record[key]
            if (index + 1) % 10 == 0:
                n = tallies["ok"]
                minutes = (time.time() - started) / 60
                rates = (
                    f"plural {tallies['plural'] / n:.2f}  disjoint {tallies['disjoint'] / n:.2f}  "
                    f"nonmono {tallies['nonmono'] / n:.2f}"
                    if n
                    else "no evaluable cases yet"
                )
                print(f"[probe {index + 1}/{len(wrong)}  {minutes:.0f} min] {rates}", flush=True)

    n = tallies["ok"]
    print(f"\nwrote {out}")
    print(f"probed {sum(tallies[s] for s in ('ok', 'parametric', 'budget'))} wrong cases at k={args.k}: "
          f"evaluable {n}, parametric {tallies['parametric']}, budget-exhausted {tallies['budget']}")
    if n:
        print(f"  plural   (two distinct certified sets) >= {tallies['plural'] / n:.2f}   [k=6 grid: 0.45-0.63]")
        print(f"  disjoint (the two share no passage)    >= {tallies['disjoint'] / n:.2f}   [k=6 grid: 0.28]")
        print(f"  nonmono  (a superset breaks it)        >= {tallies['nonmono'] / n:.2f}   [k=6 grid: 0.67-0.87]")
        print(f"  certified set size: mean {sum(sizes) / len(sizes):.2f}  "
              f"histogram {dict(sorted(Counter(sizes).items()))}")
        print(f"  queries per case: mean {sum(queries) / len(queries):.0f}")
        print("Rates at or above the k=6 values mean the structure persists at deployment depth; "
              "a sharp drop would mark it as a small-context artifact. The bounded search "
              "undercounts by construction, so whatever it finds is conservative.")


if __name__ == "__main__":
    main()
