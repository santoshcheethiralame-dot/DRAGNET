"""Remove the smallest sufficient set and see what happens — necessity and repair, one pass.

The families are sufficiency-certified; this certifies the dual and measures the utility. For
every wrong case with an enumerated family, generate once with the smallest member removed from
the context: did the answer change (the set was necessary), and did it become the gold answer
(removal REPAIRS the error — the number a debugger acting on the dragnet actually gets)?
Answer equivalence reuses the benchmark's answer_key, same as the game. One generation per case;
runs on a local GPU or any chat API. Resumable by qid.

    python scripts/run_removal.py --cell runs/hotpotqa/natural/cerebras-gpt-oss-120b \\
        --model gpt-oss-120b --provider cerebras
"""
import argparse
import json
from pathlib import Path

from lineup.data.serialization import read_generations, read_scenarios
from lineup.oracle import answer_key
from lineup.prompt import build_messages_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--provider", default=None, help="run over an API instead of a local GPU")
    parser.add_argument("--min-interval", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    scenarios = {s.qid: s for s in read_scenarios(args.cell / "scenarios.jsonl")}
    wrong = [g for g in read_generations(args.cell / "generations.jsonl") if not g.is_correct]
    if args.limit:
        wrong = wrong[: args.limit]
    families = {}
    for line in (args.cell / "mscs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            members = [frozenset(s) for s in record["minimal_sufficient"] if s]
            if not record["parametric"] and members:
                families[record["qid"]] = min(members, key=lambda m: (len(m), sorted(m)))

    if args.provider:
        from lineup.backends.api_backend import PROVIDERS, APIModel

        model = APIModel(args.model, base_url=PROVIDERS[args.provider],
                         max_new_tokens=args.max_tokens, min_interval=args.min_interval)
    else:
        from lineup.backends import TransformersModel

        model = TransformersModel(args.model, max_new_tokens=32, load_in_4bit=args.load_in_4bit)

    out = args.cell / "removal.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["qid"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    with out.open("a", encoding="utf-8") as handle:
        for index, generation in enumerate(wrong):
            member = families.get(generation.qid)
            scenario = scenarios.get(generation.qid)
            if member is None or scenario is None or generation.qid in done:
                continue
            kept = [c for c in scenario.chunks if c.chunk_id not in member]
            answer = model.generate(build_messages_for(scenario.question, kept)).text.strip()
            gold, planted = scenario.gold_answer, scenario.recipe.intended_wrong_answer
            before = answer_key(generation.model_answer, gold, planted)
            after = answer_key(answer, gold, planted)
            handle.write(json.dumps({
                "qid": generation.qid,
                "removed": sorted(member),
                "answer_after": answer,
                "changed": after != before,
                "repaired": after == answer_key(gold, gold, planted),
            }, ensure_ascii=False) + "\n")
            handle.flush()
            if (index + 1) % 10 == 0:
                print(f"[{index + 1}/{len(wrong)}]", flush=True)

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    print(f"\nwrote {out}  (n={n})")
    if n:
        print(f"necessity (answer changed): {sum(r['changed'] for r in rows) / n:.2f}")
        print(f"repair (answer became gold): {sum(r['repaired'] for r in rows) / n:.2f}")


if __name__ == "__main__":
    main()
