"""Sensitivity of the family-structure rates to the answer-equivalence granularity (reviewer R1.3).

Every headline rate is defined relative to "the model on a subset reproduces an answer equivalent to the
wrong answer a," where equivalence is the benchmark's ``answer_key`` map to {gold, wrong, residual}. The
salience axis carries a three-way strict/phrase/loose sensitivity table (main paper); the equivalence
axis did not. The disjoint co-sufficiency rate is the most exposed: two disjoint subsets count as
co-sufficient only if both land in the same wrong class, so any coarseness in that class inflates it.

This probe brackets the shipped matcher with a stricter and a looser one and reports H1, plurality, and
disjointness under each, so a reader can see whether the structure survives a strict reading:

  exact      : normalize(answer) == normalize(a)                       (identical string; strictest)
  canonical  : answer_key(answer) == answer_key(a)                     (shipped: phrase on gold/wrong,
                                                                         else normalized residual)
  cluster    : canonical, or -- for residual-class answers only -- the two normalized token sets nest
               or overlap at Jaccard >= 0.6                             (aliases/abbreviations; loosest)

The three nest (exact subset of canonical subset of cluster), so the rates are monotone and any collapse
at the strict end is visible. One model pass caches the raw generation for every bounded subset; the
three equivalences are then scored offline over the identical query set, and the cache is written so a
further granularity re-scores with no GPU.

    python scripts/run_equivalence_sweep.py --cell runs/hotpotqa/natural/qwen \\
        --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --max-size 3 --limit 120
"""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import torch

from lineup.config import DEFAULT_MODEL, DEFAULT_SEED, set_seed
from lineup.data.serialization import read_generations, read_scenarios
from lineup.oracle import answer_key
from lineup.prompt import build_messages_for
from lineup.textnorm import normalize

from dragnet.game import Game
from dragnet.mscs import minimal_sufficient_sets


def subset_messages(scenario, subset):
    shown = [c for c in scenario.chunks if c.chunk_id in subset]
    return build_messages_for(scenario.question, shown)


def eq_exact(answer, target, gold, wrong):
    return normalize(answer) == normalize(target)


def eq_canonical(answer, target, gold, wrong):
    return answer_key(answer, gold, wrong) == answer_key(target, gold, wrong)


def eq_cluster(answer, target, gold, wrong):
    if eq_canonical(answer, target, gold, wrong):
        return True
    ka, kt = answer_key(answer, gold, wrong), answer_key(target, gold, wrong)
    if ka in ("gold", "wrong") or kt in ("gold", "wrong"):
        return False                                   # a known-value answer never loosely matches another value
    a, t = set(normalize(answer).split()), set(normalize(target).split())
    if not a or not t:
        return False
    return a <= t or t <= a or len(a & t) / len(a | t) >= 0.6


EQUIVALENCES = {"exact": eq_exact, "canonical": eq_canonical, "cluster": eq_cluster}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cache_subsets(model, scenario, max_size):
    """Generate once for every subset up to the bound; return {frozenset(ids): raw_text} or None on OOM."""
    ids = [c.chunk_id for c in scenario.chunks]
    limit = min(max_size, len(ids))
    cache = {}
    for size in range(0, limit + 1):
        for combo in combinations(ids, size):
            s = frozenset(combo)
            try:
                cache[s] = model.generate(subset_messages(scenario, s)).text.strip()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return None, ids
    return cache, ids


def score(cache, ids, target, gold, wrong, max_size):
    """Family structure under each equivalence, read off the cached generations (no model calls)."""
    out = {}
    for name, eq in EQUIVALENCES.items():
        game = Game(ids, lambda s, e=eq: e(cache.get(frozenset(s), ""), target, gold, wrong))
        mss = minimal_sufficient_sets(game, max_size)
        nonempty = [s for s in mss if s]
        out[name] = {
            "parametric": frozenset() in mss,
            "n_mss": len(nonempty),
            "h1": bool(nonempty),
            "plural": len(nonempty) >= 2,
            "disjoint": any(not (a & b) for a, b in combinations(nonempty, 2)),
        }
    return out


def report(rows, key):
    """Rate of a boolean property among cases evaluable (non-parametric) under `key`'s equivalence."""
    ev = [r[key] for r in rows if not r[key]["parametric"]]
    n = len(ev)
    line = {"n": n}
    for prop in ("h1", "plural", "disjoint"):
        k = sum(r[prop] for r in ev)
        lo, hi = wilson(k, n)
        line[prop] = (k / n if n else 0.0, lo, hi)
    return line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=Path, required=True, help="natural cell with scenarios.jsonl + generations.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--max-size", type=int, default=3, help="enumeration bound for minimal sufficient sets")
    ap.add_argument("--limit", type=int, default=120, help="wrong cases to probe (0 = all)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    set_seed(args.seed)

    from lineup.backends import TransformersModel

    model = TransformersModel(args.model, max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit)
    scenarios = {s.qid: s for s in read_scenarios(args.cell / "scenarios.jsonl")}
    wrong = [g for g in read_generations(args.cell / "generations.jsonl") if not g.is_correct]
    if args.limit:
        wrong = wrong[: args.limit]

    cache_out = (args.cell / "equiv_cache.jsonl").open("w", encoding="utf-8")
    rows, skipped = [], 0
    for i, generation in enumerate(wrong):
        scenario = scenarios.get(generation.qid)
        if scenario is None:
            continue
        cache, ids = cache_subsets(model, scenario, args.max_size)
        if cache is None:
            skipped += 1
            continue
        gold = scenario.gold_answer
        wrong_val = scenario.recipe.intended_wrong_answer
        verdicts = score(cache, ids, generation.model_answer, gold, wrong_val, args.max_size)
        rows.append(verdicts)
        cache_out.write(json.dumps({
            "qid": generation.qid, "gold": gold, "wrong": wrong_val, "target": generation.model_answer,
            "ids": ids, "cache": {"|".join(sorted(s)): t for s, t in cache.items()},
        }) + "\n")
        if (i + 1) % 10 == 0:
            d = report(rows, "canonical")
            print(f"[{i + 1}/{len(wrong)}] n={d['n']}  canonical disjoint {d['disjoint'][0]:.2f}"
                  f"  h1 {d['h1'][0]:.2f}  (oom-skipped {skipped})", flush=True)
    cache_out.close()

    out = args.cell / "equiv_sweep.json"
    summary = {name: report(rows, name) for name in EQUIVALENCES}
    shared = [r for r in rows if all(not r[name]["parametric"] for name in EQUIVALENCES)]
    out.write_text(json.dumps({"per_equivalence": summary, "n_cases": len(rows),
                               "n_shared_evaluable": len(shared), "oom_skipped": skipped}, indent=2))

    print(f"\nwrote {out}  (cases {len(rows)}, oom-skipped {skipped}, cache {args.cell / 'equiv_cache.jsonl'})")
    print(f"{'equivalence':>10} | {'n':>4} | {'H1':>16} | {'plural':>16} | {'disjoint':>16}")
    for name in EQUIVALENCES:
        d = summary[name]
        def cell(x):
            return f"{x[0]:.2f} [{x[1]:.2f},{x[2]:.2f}]"
        print(f"{name:>10} | {d['n']:>4} | {cell(d['h1']):>16} | {cell(d['plural']):>16} | {cell(d['disjoint']):>16}")
    print("Read: disjoint and H1 monotone across strict->loose; if disjoint holds at 'exact', the "
          "marquee rate is not an artifact of a coarse wrong-answer class.")


if __name__ == "__main__":
    main()
