"""Robustness of non-monotonicity and disjointness to greedy near-ties (reviewer R1.1).

A skeptic reads "adding a passage breaks reproduction" as a passage nudging a logit near-tie across
the argmax boundary, not as content that genuinely redirects the hop. This probe separates the two on
the natural cell, two ways, per violating add S -> S u {p} (S reproduces the wrong answer a, S u {p}
does not under greedy decoding):

  margin   : the drop in the wrong answer's total log-probability, logprob_a(S) - logprob_a(S u {p}).
             A near-tie flip has a small drop; a meaningful redirection has a large one. We report the
             non-monotonicity rate as a function of a minimum-drop threshold, so a reader can see how
             much of it survives when marginal flips are excluded.
  resampled: re-verify the flip under temperature sampling -- S must still reproduce a on a majority of
             N samples and S u {p} must still break it -- so a flip that only holds for the greedy
             argmax is not counted. We report the non-monotonicity (and disjointness) rate under this
             resampled predicate beside the greedy one.

Disjointness is re-checked the same way: each of the two disjoint sufficient sets must reproduce a on a
majority of N samples to count. Everything else -- the reproduction predicate, the enumeration -- is the
paper's, so the numbers are directly comparable to the main grid.

    python scripts/run_robust_probe.py --cell runs/hotpotqa/natural/qwen \\
        --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --limit 120
"""
import argparse
import json
import statistics
from itertools import combinations
from pathlib import Path

import torch

from lineup.config import DEFAULT_MODEL, DEFAULT_SEED, set_seed
from lineup.data.serialization import read_generations, read_scenarios
from lineup.oracle import answer_key
from lineup.prompt import build_messages_for

from dragnet.model_game import scenario_game
from dragnet.mscs import is_sufficient, minimal_sufficient_sets


def subset_messages(scenario, subset):
    shown = [c for c in scenario.chunks if c.chunk_id in subset]
    return build_messages_for(scenario.question, shown)


def answer_logprob(model, scenario, subset, answer):
    """Total log-probability the model assigns to the wrong answer a, shown only `subset`."""
    scoring = model.score(subset_messages(scenario, subset), answer)
    return float(sum(scoring.logprobs)) if scoring.logprobs else 0.0


@torch.no_grad()
def sampled_reproduce_frac(model, scenario, subset, target_key, gold, wrong, n, temperature):
    """Fraction of n temperature samples from `subset` that reproduce the wrong answer's class."""
    payload = [{"role": m.role, "content": m.content} for m in subset_messages(scenario, subset)]
    ids = model.tokenizer.apply_chat_template(payload, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    ids = ids.to(model.device)
    out = model.model.generate(
        ids, do_sample=True, temperature=temperature, num_return_sequences=n,
        max_new_tokens=model.max_new_tokens,
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
    )
    hits = 0
    for seq in out:
        text = model.tokenizer.decode(seq[ids.shape[1]:], skip_special_tokens=True).strip()
        hits += int(answer_key(text, gold, wrong) == target_key)
    return hits / n


def robust_case(model, scenario, generation, max_size, n_samples, temperature, sample_thresh):
    """Return the greedy / margin / resampled verdicts for one wrong case, or None if not evaluable."""
    game = scenario_game(model, scenario, generation.model_answer)
    mss = [s for s in minimal_sufficient_sets(game, max_size=max_size) if s]  # drop the empty (parametric)
    if not mss:
        return None
    gold = scenario.gold_answer
    wrong = scenario.recipe.intended_wrong_answer
    target = answer_key(generation.model_answer, gold, wrong)

    ids = list(game.ids)
    margins, robust_violation = [], False
    greedy_violation = False
    for s in mss:
        base_lp = None
        for p in (c for c in ids if c not in s):
            sup = s | {p}
            if is_sufficient(game, sup):
                continue                                   # superset still reproduces: no violation
            greedy_violation = True
            if base_lp is None:
                base_lp = answer_logprob(model, scenario, s, generation.model_answer)
            drop = base_lp - answer_logprob(model, scenario, sup, generation.model_answer)
            margins.append(drop)
            # resampled: S must still reproduce, the superset must still break, under sampling
            rep_s = sampled_reproduce_frac(model, scenario, s, target, gold, wrong, n_samples, temperature)
            rep_sup = sampled_reproduce_frac(model, scenario, sup, target, gold, wrong, n_samples, temperature)
            if rep_s >= sample_thresh and rep_sup <= (1.0 - sample_thresh):
                robust_violation = True

    # disjoint: any two MSS sharing no passage; require both to reproduce under sampling to count
    disjoint_greedy = disjoint_robust = False
    for a, b in combinations(mss, 2):
        if a & b:
            continue
        disjoint_greedy = True
        ra = sampled_reproduce_frac(model, scenario, a, target, gold, wrong, n_samples, temperature)
        rb = sampled_reproduce_frac(model, scenario, b, target, gold, wrong, n_samples, temperature)
        if ra >= sample_thresh and rb >= sample_thresh:
            disjoint_robust = True
            break

    return {
        "qid": generation.qid,
        "n_mss": len(mss),
        "nonmono_greedy": greedy_violation,
        "nonmono_robust": robust_violation,
        "disjoint_greedy": disjoint_greedy,
        "disjoint_robust": disjoint_robust,
        "margins": margins,
        "queries": game.queries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=Path, required=True, help="natural cell with scenarios.jsonl + generations.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=120, help="wrong cases to probe (0 = all)")
    ap.add_argument("--max-size", type=int, default=3, help="enumeration bound for minimal sufficient sets")
    ap.add_argument("--n-samples", type=int, default=5, help="temperature samples per subset")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--sample-thresh", type=float, default=0.6, help="majority fraction to count a robust reproduce/break")
    ap.add_argument("--margin-thresholds", default="0.5,1.0,2.0", help="min log-prob drop (nats) to count a meaningful flip")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    set_seed(args.seed)

    from lineup.backends import TransformersModel

    model = TransformersModel(args.model, max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit)
    scenarios = {s.qid: s for s in read_scenarios(args.cell / "scenarios.jsonl")}
    wrong = [g for g in read_generations(args.cell / "generations.jsonl") if not g.is_correct]
    if args.limit:
        wrong = wrong[: args.limit]
    thresholds = [float(x) for x in args.margin_thresholds.split(",")]

    out = args.cell / "robust_probe.jsonl"
    rows = []
    with out.open("w", encoding="utf-8") as handle:
        for i, generation in enumerate(wrong):
            scenario = scenarios.get(generation.qid)
            if scenario is None:
                continue
            try:
                rec = robust_case(model, scenario, generation, args.max_size, args.n_samples,
                                  args.temperature, args.sample_thresh)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            if rec is None:
                continue
            handle.write(json.dumps(rec) + "\n")
            rows.append(rec)
            if (i + 1) % 10 == 0:
                nm = sum(r["nonmono_greedy"] for r in rows) / len(rows)
                nmr = sum(r["nonmono_robust"] for r in rows) / len(rows)
                print(f"[{i + 1}/{len(wrong)}] evaluable {len(rows)}  nonmono greedy {nm:.2f}  robust {nmr:.2f}", flush=True)

    n = len(rows)
    print(f"\nwrote {out}  (evaluable {n})")
    if not n:
        return
    all_margins = [m for r in rows for m in r["margins"]]
    print(f"non-monotonicity  greedy {sum(r['nonmono_greedy'] for r in rows) / n:.2f}"
          f"   resampled (>={args.sample_thresh:.0%} of {args.n_samples}) {sum(r['nonmono_robust'] for r in rows) / n:.2f}")
    for t in thresholds:
        rate = sum(any(m > t for m in r["margins"]) for r in rows) / n
        print(f"  non-monotone with a log-prob drop > {t} nats: {rate:.2f}")
    if all_margins:
        qs = statistics.quantiles(all_margins, n=4) if len(all_margins) >= 4 else [float('nan')] * 3
        print(f"  violating-add log-prob drop (nats): median {statistics.median(all_margins):.2f}"
              f"  IQR [{qs[0]:.2f}, {qs[2]:.2f}]  n_adds {len(all_margins)}")
    dg = sum(r["disjoint_greedy"] for r in rows) / n
    dr = sum(r["disjoint_robust"] for r in rows) / n
    print(f"disjoint sufficient sets  greedy {dg:.2f}   resampled {dr:.2f}")
    print("Read: if the resampled and high-margin rates stay close to the greedy rate, the structure is "
          "not greedy near-tie noise; a large gap would mean much of it is decision-boundary fragility.")


if __name__ == "__main__":
    main()
