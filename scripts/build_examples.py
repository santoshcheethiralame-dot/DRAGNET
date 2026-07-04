"""Pull the cleanest worked example of each error fate from real cells — the paper's specimen box.

Four fates, one auto-picked candidate each (shortest question, smallest context footprint, so it
fits a figure box): a clean CULPRIT (one singleton member), a DISJOINT COALITION (two sufficient
explanations sharing no passage — the identifiability jewel), a HOLISTIC error (nothing inside
the bound suffices, only the whole context), and a PARAMETRIC case (the empty context already
reproduces the answer — contamination made visible). Writes paper/examples.md with several
candidates per fate; the author hand-picks and trims for the paper.

    python scripts/build_examples.py --cells <natural cells...> --out paper/examples.md
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

from lineup.data.serialization import read_generations, read_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("paper/examples.md"))
    parser.add_argument("--per-fate", type=int, default=3)
    args = parser.parse_args()

    fates: dict[str, list] = {"culprit": [], "disjoint coalition": [], "holistic": [], "parametric": []}
    for cell in args.cells:
        scenarios = {s.qid: s for s in read_scenarios(cell / "scenarios.jsonl")}
        answers = {g.qid: g.model_answer for g in read_generations(cell / "generations.jsonl") if not g.is_correct}
        for line in (cell / "mscs.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            scenario = scenarios.get(row["qid"])
            if scenario is None:
                continue
            members = [frozenset(s) for s in row["minimal_sufficient"] if s]
            if row["parametric"]:
                fate = "parametric"
            elif not members:
                fate = "holistic"
            elif len(members) == 1 and len(members[0]) == 1:
                fate = "culprit"
            elif any(not (a & b) for a, b in combinations(members, 2)):
                fate = "disjoint coalition"
            else:
                continue   # ordinary overlapping coalitions — plenty, not specimen-grade
            fates[fate].append((len(scenario.question), cell, row, scenario, answers.get(row["qid"], "")))

    lines = [
        "# Worked examples — auto-picked candidates per fate",
        "",
        "Hand-pick one per fate for the paper box; trim passage text as needed. Sorted by",
        "question length (shortest first) so the cleanest specimens lead.",
    ]
    for fate, candidates in fates.items():
        lines += ["", f"## {fate}  ({len(candidates)} candidates)"]
        for _, cell, row, scenario, answer in sorted(candidates, key=lambda x: x[0])[: args.per_fate]:
            titles = {c.chunk_id: c.title for c in scenario.chunks}
            family = " | ".join(
                "{" + ", ".join(titles.get(x, x) for x in sorted(s)) + "}" for s in row["minimal_sufficient"] if s
            ) or "(none within bound)"
            lines += [
                "",
                f"**{'/'.join(cell.parts[-3:])} · {row['qid']}**",
                f"- Q: {scenario.question}",
                f"- gold: {scenario.gold_answer}   ·   model answered: {answer}",
                f"- sufficient family: {family}",
            ]
            member_ids = set().union(*[frozenset(s) for s in row["minimal_sufficient"] if s]) if row["minimal_sufficient"] else set()
            for chunk in scenario.chunks:
                if chunk.chunk_id in member_ids:
                    text = chunk.text[:220] + ("..." if len(chunk.text) > 220 else "")
                    lines.append(f"  - [{chunk.title}] {text}")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}  " + "  ".join(f"{k}:{len(v)}" for k, v in fates.items()))


if __name__ == "__main__":
    main()
