"""Walk a tree of run cells and regenerate paper/results.md — every table, one command.

Reads whatever artifacts each cell has and fills what it can: the natural grid (H1, ambiguity,
A3, minimum-size histogram), the conformal table per ranking arm, and the designed-arm verdicts
(H3a/H3b/H4). Cells missing an artifact show as pending rather than being dropped, so the file
is an honest inventory of the evidence as well as its summary.

    python scripts/build_results.py --root path/to/runs --out paper/results.md
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from lineup.data.serialization import read_roles, read_scenarios

from _cells import load_cases
from scope.conformal import depth_item
from scope.prereg import (
    h1_small_sets,
    h2_coverage,
    h3_designed_sufficient,
    h3_link_silent,
    h4_paired_arms,
    h5_loo_intersection,
)

ARMS = (("contextcite", "ranking"), ("interaction", "interaction"), ("shapley", "shapley"))


def _rows(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt(value, places=2):
    return f"{value:.{places}f}" if value is not None else "—"


def _ci(pair):
    low, high = pair
    return f" [{low:.2f}, {high:.2f}]" if low is not None else ""


def natural_table(cells: list[Path], n_boot: int) -> list[str]:
    lines = [
        "## Natural grid — H1, ambiguity, A3",
        "",
        "| cell | rows | bound | parametric | H1 small-set | ambiguity | A3 rate | min-size hist |",
        "|---|--:|--:|--:|---|--:|--:|---|",
    ]
    for cell in cells:
        rows = _rows(cell / "mscs.jsonl")
        label = "/".join(cell.parts[-3:])
        if rows is None:
            lines.append(f"| {label} | — | — | — | pending | — | — | — |")
            continue
        h1 = h1_small_sets(rows, n_boot=n_boot)
        nonparam = [r for r in rows if not r["parametric"]]
        ambiguous = sum(1 for r in nonparam if len([s for s in r["minimal_sufficient"] if s]) > 1)
        violating = sum(1 for r in rows if r["monotonicity_violations"])
        sizes = Counter(
            min(len(s) for s in r["minimal_sufficient"])
            for r in nonparam
            if r["minimal_sufficient"]
        )
        hist = " ".join(f"{k}:{v}" for k, v in sorted(sizes.items()))
        lines.append(
            f"| {label} | {len(rows)} | {rows[0]['max_size']} | {_fmt(h1['parametric_fraction'])} "
            f"| {_fmt(h1['rate'])}{_ci(h1['interval'])} | {ambiguous / len(nonparam):.2f} "
            f"| {violating / len(rows):.2f} | {hist} |"
        )
    return lines


def conformal_table(cells: list[Path], alphas: tuple[float, ...], seeds: tuple[int, ...]) -> list[str]:
    lines = [
        "## Conformal coverage per ranking arm (seeded halves, all seeds shown as min–max)",
        "",
        "| cell | arm | alpha | tau | coverage | mean size | n/test |",
        "|---|---|--:|---|---|---|--:|",
    ]
    pooled = {arm: [] for arm, _ in ARMS}
    for cell in cells:
        if not (cell / "mscs.jsonl").exists():
            continue
        label = "/".join(cell.parts[-3:])
        cases = load_cases(cell, "mscs")
        for arm, attr in ARMS:
            items = [depth_item(c.key, getattr(c, attr), c.family) for c in cases if getattr(c, attr)]
            if not items:
                continue
            pooled[arm].extend(items)
            for alpha in alphas:
                splits = [s for s in h2_coverage(items, alpha=alpha, seeds=seeds) if "coverage" in s]
                if not splits:
                    continue
                taus = sorted({("inf" if s["tau"] == float("inf") else f"{s['tau']:.0f}") for s in splits})
                covs = [s["coverage"] for s in splits]
                sizes = [s["mean_size"] for s in splits]
                lines.append(
                    f"| {label} | {arm} | {alpha} | {'/'.join(taus)} "
                    f"| {min(covs):.2f}–{max(covs):.2f} | {min(sizes):.1f}–{max(sizes):.1f} | {splits[0]['n_test']} |"
                )
    for arm, items in pooled.items():
        if not items:
            continue
        for alpha in alphas:
            splits = [s for s in h2_coverage(items, alpha=alpha, seeds=seeds) if "coverage" in s]
            if not splits:
                continue
            taus = sorted({("inf" if s["tau"] == float("inf") else f"{s['tau']:.0f}") for s in splits})
            covs = [s["coverage"] for s in splits]
            sizes = [s["mean_size"] for s in splits]
            lines.append(
                f"| **pooled** | {arm} | {alpha} | {'/'.join(taus)} "
                f"| {min(covs):.2f}–{max(covs):.2f} | {min(sizes):.1f}–{max(sizes):.1f} | {splits[0]['n_test']} |"
            )
    return lines


def designed_table(cells: list[Path], n_boot: int) -> list[str]:
    lines = [
        "## Designed arm — construction and extraction verdicts",
        "",
        "| cell | H3a designed-sufficient | H3b link vs distractor silent | H4 beam vs contextcite (designed target) |",
        "|---|---|---|---|",
    ]
    for cell in cells:
        label = "/".join(cell.parts[-3:])
        validation = _rows(cell / "validation.jsonl")
        h3a = "pending (run log)"
        if validation is not None:
            r = h3_designed_sufficient(validation, n_boot=n_boot)
            h3a = f"{_fmt(r['rate'])}{_ci(r['interval'])} (n={r['n']})"
        h3b = "pending"
        if (cell / "roles.jsonl").exists() and (cell / "scenarios.jsonl").exists():
            scenarios = {s.qid: s for s in read_scenarios(cell / "scenarios.jsonl")}
            r = h3_link_silent(read_roles(cell / "roles.jsonl"), scenarios)
            h3b = f"{_fmt(r['link_rate'])} vs {_fmt(r['distractor_rate'])} ({'PASS' if r['passed'] else 'fail'})"
        extraction = _rows(cell / "extraction.jsonl")
        h4 = "pending"
        if extraction is not None:
            r = h4_paired_arms(extraction, n_boot=max(n_boot, 2000))
            h4 = f"{_fmt(r.get('rate_a'))} vs {_fmt(r.get('rate_b'))}, p={_fmt(r.get('p'), 3)} (n={r['n']})"
        lines.append(f"| {label} | {h3a} | {h3b} | {h4} |")
    return lines


def h5_table(cells: list[Path]) -> list[str]:
    lines = [
        "## H5 — leave-one-out vs the intersection of the sufficient family",
        "",
        "| cell | agreement | clean n | violations excluded | no set in bound |",
        "|---|---|--:|--:|--:|",
    ]
    for cell in cells:
        rows = _rows(cell / "mscs.jsonl")
        if rows is None or not (cell / "roles.jsonl").exists():
            continue
        r = h5_loo_intersection(rows, read_roles(cell / "roles.jsonl"))
        label = "/".join(cell.parts[-3:])
        lines.append(f"| {label} | {_fmt(r['rate'])} | {r['n']} | {r['violating']} | {r['unevaluable']} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="tree of <dataset>/<condition>/<tag> cells")
    parser.add_argument("--out", type=Path, default=Path("paper/results.md"))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    natural = sorted(p for p in args.root.glob("*/natural/*") if (p / "scenarios.jsonl").exists())
    designed = sorted(p for p in args.root.glob("*/and/*") if (p / "scenarios.jsonl").exists())

    lines = ["# SCoPE results", "", "Regenerate: `python scripts/build_results.py --root <path-to-cells>`", ""]
    lines += natural_table(natural, args.bootstrap) + [""]
    lines += conformal_table(natural, tuple(args.alphas), tuple(args.seeds)) + [""]
    lines += h5_table(natural) + [""]
    lines += designed_table(designed, args.bootstrap)

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}  ({len(natural)} natural cells, {len(designed)} designed cells)")


if __name__ == "__main__":
    main()
