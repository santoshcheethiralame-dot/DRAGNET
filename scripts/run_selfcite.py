"""Does the model know its own sufficient sets? Self-citation vs the enumerated family.

For every wrong case, show the model the same numbered context it answered from, remind it of
its own answer, and ask which passages that answer relied on. The claim is scored against the
enumerated minimal sufficient family: does the self-citation contain a sufficient set (the
citation would survive as an attribution), does it exactly match a member, and how large is it?
The chain-of-thought faithfulness literature predicts divergence but has no ground truth to
measure it against — the family is exactly that ground truth. Writes one selfcite.jsonl row per
case; prints the comparison when the cell has an mscs.jsonl.

    python scripts/run_selfcite.py --cell runs/hotpotqa/natural/qwen \\
        --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
"""
import argparse
import json
import re
from pathlib import Path

from lineup.backends.base import Message
from lineup.data.serialization import read_generations, read_scenarios

from dragnet.designed import set_covers


def citation_prompt(scenario, answer: str) -> list[Message]:
    passages = "\n\n".join(
        f"[{i + 1}] {chunk.title}: {chunk.text}" for i, chunk in enumerate(scenario.chunks)
    )
    text = (
        f"Context passages:\n\n{passages}\n\n"
        f"Question: {scenario.question}\n"
        f"You answered: {answer}\n\n"
        "Which passages did that answer rely on? Reply with only the passage numbers, "
        "comma-separated, smallest sufficient selection only."
    )
    return [Message("user", text)]


def parse_citations(reply: str, scenario) -> list[str]:
    numbers = {int(n) for n in re.findall(r"\d+", reply)}
    return [scenario.chunks[n - 1].chunk_id for n in sorted(numbers) if 1 <= n <= len(scenario.chunks)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--provider", default=None, help="run over an API instead of a local GPU (e.g. cerebras)")
    parser.add_argument("--min-interval", type=float, default=2.0, help="seconds between API calls")
    parser.add_argument("--max-tokens", type=int, default=256, help="API output budget (reasoning models think first)")
    parser.add_argument("--limit", type=int, default=0, help="wrong cases (0 = all)")
    args = parser.parse_args()

    scenarios = {s.qid: s for s in read_scenarios(args.cell / "scenarios.jsonl")}
    wrong = [g for g in read_generations(args.cell / "generations.jsonl") if not g.is_correct]
    if args.limit:
        wrong = wrong[: args.limit]

    if args.provider:
        from lineup.backends.api_backend import PROVIDERS, APIModel

        model = APIModel(args.model, base_url=PROVIDERS[args.provider],
                         max_new_tokens=args.max_tokens, min_interval=args.min_interval)
    else:
        from lineup.backends import TransformersModel

        model = TransformersModel(args.model, max_new_tokens=48, load_in_4bit=args.load_in_4bit)

    out = args.cell / "selfcite.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for index, generation in enumerate(wrong):
            scenario = scenarios.get(generation.qid)
            if scenario is None:
                continue
            reply = model.generate(citation_prompt(scenario, generation.model_answer)).text
            handle.write(json.dumps({
                "qid": generation.qid,
                "reply": reply,
                "cited": parse_citations(reply, scenario),
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 10 == 0:
                print(f"[{index + 1}/{len(wrong)}]", flush=True)
    print(f"wrote {out}")

    mscs_path = args.cell / "mscs.jsonl"
    if not mscs_path.exists():
        print("no mscs.jsonl in the cell — run the enumeration to score the citations")
        return
    families = {}
    for line in mscs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if not record["parametric"]:
                families[record["qid"]] = [frozenset(s) for s in record["minimal_sufficient"] if s]
    n = covers = exact = 0
    sizes = []
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        family = families.get(row["qid"])
        if not family:
            continue
        cited = frozenset(row["cited"])
        n += 1
        covers += set_covers(cited, family)
        exact += cited in family
        sizes.append(len(cited))
    if n:
        print(f"\nself-citation vs enumerated family (n={n}):")
        print(f"  contains a sufficient set: {covers / n:.2f}")
        print(f"  exactly a minimal member:  {exact / n:.2f}")
        print(f"  mean citation size:        {sum(sizes) / n:.1f}")


if __name__ == "__main__":
    main()
