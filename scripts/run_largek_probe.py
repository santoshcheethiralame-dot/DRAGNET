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

    python scripts/run_largek_probe.py --cell runs/hotpotqa/largek20/qwen \\
        --dataset hotpotqa --k 20 --limit 400 \\
        --model Qwen/Qwen2.5-7B-Instruct --device-map auto

On a 16 GB card, run 4-bit (--load-in-4bit): the weights are ~5.5 GB, leaving room for the
~4-6 GB a k=20 prefill needs, and it is far faster than an fp16 model sharded across two cards
(which pays a cross-device transfer at every layer). k=30 overflows even 4-bit here, so k=20 is
the depth a single T4 sustains; deeper contexts want a larger GPU. Each per-case generation is
guarded against OOM so one long-tail context is skipped, not fatal.
"""
import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

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


@dataclass
class _Gen:
    text: str
    mean_logprob: float = 0.0
    truncated: bool = False


class VLLMModel:
    """vLLM backend for large k on a single 16 GB card. Paged KV and chunked prefill bound the
    memory of a long-context prompt, and AWQ 4-bit weights (~4.5 GB) leave the rest for the cache,
    so k = 50-100 fits where transformers OOMs. Exposes the same .generate(messages) -> object with
    .text that the probe and the lineup build path expect, so nothing downstream changes."""

    def __init__(self, model_name, *, max_new_tokens=24, max_model_len=8192, quantization="awq",
                 gpu_memory_utilization=0.90):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(model=model_name, quantization=quantization, dtype="float16",
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       enforce_eager=True)
        self.default = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        self.max_new_tokens = max_new_tokens

    def _prompt(self, messages):
        payload = [{"role": m.role, "content": m.content} for m in messages]
        return self.tokenizer.apply_chat_template(payload, add_generation_prompt=True, tokenize=False)

    def generate(self, messages, max_new_tokens=None):
        sp = self.default if max_new_tokens is None else self._SamplingParams(
            temperature=0.0, max_tokens=max_new_tokens)
        try:
            out = self.llm.generate([self._prompt(messages)], sp, use_tqdm=False)
        except Exception as exc:
            # A prompt longer than max_model_len is unevaluable; re-raise as OOM so the per-case
            # guards skip it rather than crash the run (raise max_model_len to avoid it entirely).
            if "context length" in str(exc) or "input tokens" in str(exc):
                raise torch.cuda.OutOfMemoryError(str(exc))
            raise
        return _Gen(text=out[0].outputs[0].text)


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

    # Only the probe's wrong-case budget is needed, so stop once a small surplus is in hand (some
    # wrong cases turn out parametric and leave the evaluable pool) instead of generating on every
    # one of `limit` questions -- at large k each generation is slow, and the build otherwise
    # dominates the run.
    target_wrong = args.limit_cases + 15 if args.limit_cases else None

    scenarios, generations, skipped, n_wrong = [], [], 0, 0
    for index, example in enumerate(examples):
        scenario = builder.build(example)
        if scenario is None:
            skipped += 1
            continue
        try:
            generation = generate_and_judge(model, scenario, llm_judge=judge)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()            # a rare over-long context; skip it, keep the run alive
            skipped += 1
            continue
        scenarios.append(scenario)
        generations.append(generation)
        n_wrong += int(not generation.is_correct)
        if (index + 1) % 25 == 0:
            print(f"[build {index + 1}/{len(examples)}] cases {len(scenarios)}  wrong {n_wrong}", flush=True)
        if target_wrong and n_wrong >= target_wrong:
            print(f"[build] {n_wrong} wrong cases in hand (target {target_wrong}); stopping build", flush=True)
            break

    write_scenarios(args.cell / "scenarios.jsonl", scenarios)
    write_generations(args.cell / "generations.jsonl", generations)
    print(f"built {len(scenarios)} cases at k={args.k} (skipped {skipped}), wrong {n_wrong}", flush=True)
    return scenarios, generations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--dataset", default="hotpotqa", help="hotpotqa, 2wiki, or musique")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=400, help="source questions to draw")
    parser.add_argument("--k", type=int, default=20,
                        help="retrieval depth of the built contexts (20 is the dual-T4 ceiling; "
                             "a 30-passage prefill OOMs even sharded fp16)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--device-map", default=None,
        help="shard an fp16 model across the visible GPUs (e.g. 'auto' for two T4s); takes "
             "precedence over --load-in-4bit. Needed at large k, where a 30-passage prefill "
             "overflows a single 16 GB card in 4-bit.",
    )
    parser.add_argument(
        "--backend", default="transformers", choices=["transformers", "vllm"],
        help="'vllm' uses paged KV + chunked prefill (AWQ weights) to fit k=50-100 on a 16 GB card, "
             "where the transformers path OOMs on the long prefill.",
    )
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vLLM context length; must exceed the prompt (k=50 ~6k tokens, k=100 ~12k).")
    parser.add_argument("--quantization", default="awq",
                        help="vLLM weight quantization: awq or gptq (needs a matching checkpoint), or none.")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit-cases", type=int, default=120, help="wrong cases to probe (0 = all)")
    parser.add_argument("--nonmono-samples", type=int, default=4, help="sampled supersets per case")
    parser.add_argument("--budget", type=int, default=0, help="query budget per extraction (0 = 2k+8)")
    args = parser.parse_args()

    set_seed(args.seed)
    from lineup.backends import TransformersModel

    if args.backend == "vllm":
        # Paged KV + chunked prefill fit the long k=50-100 prompt a 16 GB card cannot hold in
        # transformers; AWQ weights keep the precision change immaterial (Section 8's 4-bit check).
        model = VLLMModel(args.model, max_new_tokens=args.max_new_tokens,
                          max_model_len=args.max_model_len,
                          quantization=(None if args.quantization == "none" else args.quantization))
    elif args.device_map:
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
            try:
                game = scenario_game(model, scenario, generation.model_answer)
                record = bounded_probe(game, rng, args.nonmono_samples, budget)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()        # over-long context: record and move on, don't die
                record = {"status": "oom", "queries": 0}
            handle.write(json.dumps({"qid": generation.qid, "model_answer": generation.model_answer, **record}) + "\n")

            tallies[record["status"]] += 1
            queries.append(record["queries"])
            if record["status"] == "ok":
                sizes.append(len(record["mss1"]))
                for key in ("plural", "disjoint", "nonmono"):
                    tallies[key] += record[key]
            if (index + 1) % 10 == 0:
                torch.cuda.empty_cache()        # keep fragmentation flat across thousands of subset queries
            if (index + 1) % 2 == 0:            # frequent progress: large-k cases are slow, don't run blind
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
    print(f"probed {sum(tallies[s] for s in ('ok', 'parametric', 'budget', 'oom'))} wrong cases at k={args.k}: "
          f"evaluable {n}, parametric {tallies['parametric']}, budget-exhausted {tallies['budget']}, "
          f"oom-skipped {tallies['oom']}")
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
