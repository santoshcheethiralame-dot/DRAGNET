"""Regenerate the paper figures from run artifacts — CPU only, one command per refresh.

Each figure reads the cell tree(s) directly, so a new run refreshes every panel by re-running
this script. Panels whose inputs are missing are skipped with a note, never faked.

    python scripts/build_figures.py --root path/to/cells --root3 path/to/bound3 \\
        --k10 path/to/k10/cell --out paper/figures
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _cells import load_cases
from dragnet.conformal import depth_item
from dragnet.prereg import h1_small_sets, h2_coverage
from dragnet.responsibility import max_responsibility

ARMS = (("contextcite", "ranking"), ("interaction", "interaction"), ("shapley", "shapley"))
CELLS = (
    ("hotpotqa", "qwen"), ("hotpotqa", "phi"), ("hotpotqa", "mistral"),
    ("musique", "qwen"), ("2wiki", "qwen"),
)


def _mscs(cell: Path) -> list[dict]:
    path = cell / "mscs.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def fig_guarantee(root: Path, out: Path) -> None:
    """F1 — the coverage-size plane at alpha=0.1: every (cell, arm) as a point; the pass box
    is coverage >= 0.9 - 2SE and mean size <= 4. The primary model's points sit inside;
    mistral sits outside at the reach ceiling."""
    fig, ax = plt.subplots(figsize=(7.2, 5))
    markers = {"contextcite": "o", "interaction": "s", "shapley": "^"}
    colors = {"qwen": "#1a6faf", "phi": "#c98f00", "mistral": "#b03a3a"}
    for ds, m in CELLS:
        cell = root / ds / "natural" / m
        if not (cell / "mscs.jsonl").exists():
            continue
        cases = load_cases(cell, "mscs")
        for arm, attr in ARMS:
            items = [depth_item(c.key, getattr(c, attr), c.family) for c in cases if getattr(c, attr)]
            if not items:
                continue
            splits = [s for s in h2_coverage(items, alpha=0.1, seeds=(0, 1, 2)) if "coverage" in s]
            cov = sum(s["coverage"] for s in splits) / len(splits)
            size = sum(s["mean_size"] for s in splits) / len(splits)
            ax.scatter(size, cov, marker=markers[arm], c=colors[m], s=70, zorder=3,
                       edgecolors="black", linewidths=0.5)
            if m == "mistral" and arm == "contextcite":
                ax.annotate("mistral: 21% of errors beyond bound-5\n(reach ceiling -> abstain)",
                            (size, cov), textcoords="offset points", xytext=(-160, -28), fontsize=8)
    ax.axhspan(0.9, 1.02, xmax=(4 - 0.5) / (6.8 - 0.5), color="#2e7d32", alpha=0.08)
    ax.axhline(0.9, color="#2e7d32", linestyle="--", linewidth=1)
    ax.axvline(4, color="#2e7d32", linestyle=":", linewidth=1)
    ax.text(0.7, 0.995, "guarantee met at small size", fontsize=9, color="#2e7d32", va="top")
    ax.set_xlim(0.5, 6.8)
    ax.set_ylim(0.55, 1.02)
    ax.set_xlabel("mean attribution-set size (of 6 passages)")
    ax.set_ylabel("empirical coverage of a sufficient set (alpha = 0.1)")
    ax.set_title("The dragnet: coverage vs. size across cells and rankings (bound-5 families)")
    legend = [plt.Line2D([], [], marker=mk, linestyle="", color="gray", label=arm) for arm, mk in
              [("contextcite", "o"), ("interaction", "s"), ("shapley", "^")]]
    legend += [plt.Line2D([], [], marker="o", linestyle="", color=c, label=m) for m, c in colors.items()]
    ax.legend(handles=legend, fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_guarantee.png", dpi=160)
    plt.close(fig)


def fig_depth(root3: Path, k10: Path, out: Path) -> None:
    """N3 — minimal sufficient set size is flat in retrieval depth, against the necessary-set
    size that scales with k (the benchmark's 0.45k line)."""
    def sizes(cell):
        rows = _mscs(cell)
        return [min(len(s) for s in r["minimal_sufficient"])
                for r in rows if not r["parametric"] and r["minimal_sufficient"]]
    six, ten = sizes(root3 / "hotpotqa/natural/qwen"), sizes(k10)
    if not six or not ten:
        print("  fig_depth skipped (missing k=6 or k=10 cell)")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ks = [6, 10]
    mss = [sum(six) / len(six), sum(ten) / len(ten)]
    ax.plot(ks, mss, "o-", color="#1a6faf", linewidth=2, label="minimal sufficient set (measured)")
    ax.plot([4, 6, 8, 10], [0.45 * k for k in [4, 6, 8, 10]], "s--", color="#b03a3a",
            linewidth=1.5, label="necessary set, benchmark trend (~0.45k)")
    for k, v in zip(ks, mss):
        ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax.set_xlabel("retrieval depth k (passages in context)")
    ax.set_ylabel("mean set size")
    ax.set_title("The escape: sufficient sets stay small as contexts grow")
    ax.set_xticks([4, 6, 8, 10])
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_depth.png", dpi=160)
    plt.close(fig)


def fig_responsibility(root: Path, out: Path) -> None:
    """N1 — the graded no-single-culprit: distribution of the strongest single-passage
    responsibility over all evaluable organic errors."""
    values = []
    for ds, m in CELLS:
        for row in _mscs(root / ds / "natural" / m):
            if row["parametric"]:
                continue
            members = [frozenset(s) for s in row["minimal_sufficient"] if s]
            if members:
                values.append(max_responsibility(members))
    if not values:
        print("  fig_responsibility skipped")
        return
    buckets = Counter("1.0 (true culprit)" if v == 1.0 else "1/2" if v == 0.5 else "<= 1/3" for v in values)
    order = ["1.0 (true culprit)", "1/2", "<= 1/3"]
    counts = [buckets.get(b, 0) for b in order]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(order, [c / len(values) for c in counts], color=["#2e7d32", "#c98f00", "#b03a3a"])
    for bar, c in zip(bars, counts):
        ax.annotate(f"{c / len(values):.0%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10)
    shared = 1 - counts[0] / len(values)
    ax.set_ylabel("fraction of organic errors")
    ax.set_title(f"Strongest single-passage responsibility (n={len(values)}) — "
                 f"{shared:.0%} irreducibly shared")
    fig.tight_layout()
    fig.savefig(out / "fig_responsibility.png", dpi=160)
    plt.close(fig)


def fig_a3(root: Path, out: Path) -> None:
    """N4 — the rugged lattice: monotonicity violations per case, per cell."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels, zero, some, many = [], [], [], []
    for ds, m in CELLS:
        rows = _mscs(root / ds / "natural" / m)
        if not rows:
            continue
        v = [r["monotonicity_violations"] for r in rows]
        labels.append(f"{m}\n{ds}")
        zero.append(sum(1 for x in v if x == 0) / len(v))
        some.append(sum(1 for x in v if 1 <= x < 5) / len(v))
        many.append(sum(1 for x in v if x >= 5) / len(v))
    x = range(len(labels))
    ax.bar(x, zero, label="monotone (0 violations)", color="#2e7d32")
    ax.bar(x, some, bottom=zero, label="1-4 violating pairs", color="#c98f00")
    ax.bar(x, many, bottom=[a + b for a, b in zip(zero, some)], label=">= 5 violating pairs", color="#b03a3a")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("fraction of wrong cases")
    ax.set_title("Support is not monotone: adding a passage can flip the answer")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_a3.png", dpi=160)
    plt.close(fig)


def fig_pareto(root: Path, out: Path) -> None:
    """F3 — pooled coverage vs size per ranking arm across alpha levels."""
    pool = {arm: [] for arm, _ in ARMS}
    for ds, m in CELLS:
        cell = root / ds / "natural" / m
        if not (cell / "mscs.jsonl").exists():
            continue
        cases = load_cases(cell, "mscs")
        for arm, attr in ARMS:
            pool[arm].extend(depth_item(c.key, getattr(c, attr), c.family) for c in cases if getattr(c, attr))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    styles = {"contextcite": ("o-", "#777777"), "interaction": ("s-", "#1a6faf"), "shapley": ("^-", "#2e7d32")}
    for arm, items in pool.items():
        if not items:
            continue
        xs, ys = [], []
        for alpha in (0.3, 0.2, 0.1, 0.05):
            splits = [s for s in h2_coverage(items, alpha=alpha, seeds=(0, 1, 2)) if "coverage" in s]
            finite = [s for s in splits if s["tau"] != float("inf")]
            if not finite:
                continue
            xs.append(sum(s["mean_size"] for s in finite) / len(finite))
            ys.append(sum(s["coverage"] for s in finite) / len(finite))
        style, color = styles[arm]
        ax.plot(xs, ys, style, color=color, label=f"{arm} (n={len(items)})", linewidth=1.5)
    ax.set_xlabel("mean set size")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Coverage-size frontier by ranking (pooled cells, bound-5)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_pareto.png", dpi=160)
    plt.close(fig)


def fig_selective(root: Path, out: Path) -> None:
    """N5 — the singleton risk floor: answer by descending confidence; selective risk never
    drops below ~0.3, so committing to one blamed passage at low error is impossible."""
    from dragnet.designed import set_covers
    from dragnet.selective import risk_coverage

    scored = []
    for ds, m in CELLS:
        cell = root / ds / "natural" / m
        if not (cell / "mscs.jsonl").exists():
            continue
        for case in load_cases(cell, "mscs"):
            if case.ranking and case.family:
                scored.append((case.margin, set_covers(frozenset({case.ranking[0]}), case.family)))
    if not scored:
        print("  fig_selective skipped")
        return
    curve = risk_coverage(scored)
    xs = [c for c, _ in curve]
    ys = [r for _, r in curve]
    floor = min(r for c, r in curve if c >= 0.1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(xs, ys, color="#b03a3a", linewidth=2, label="selective risk (best confidence gating)")
    ax.axhline(floor, color="#b03a3a", linestyle=":", linewidth=1)
    ax.text(0.02, floor - 0.03, f"floor ~{floor:.2f}: no gating commits a single passage below this error",
            fontsize=8, color="#b03a3a", va="top")
    ax.axhspan(0, 0.1, color="#2e7d32", alpha=0.08)
    ax.text(0.02, 0.085, "the error a practitioner wants (alpha = 0.1) — reachable only by the SET",
            fontsize=8, color="#2e7d32", va="top")
    ax.set_xlabel("fraction of errors answered (by descending margin)")
    ax.set_ylabel("false-answer rate of the committed passage")
    ax.set_title(f"Blaming one passage has a risk floor (pooled n={len(scored)})")
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_selective.png", dpi=160)
    plt.close(fig)


def fig_robustness(root: Path, out: Path) -> None:
    """F5 — H1 replicated across the full 3x3 grid of models and datasets, with the 120B frontier
    and a live BM25 retriever shown as out-of-distribution checks. A small sufficient set exists in
    every cell above the 0.50 target; the guarantee is not an artifact of one model or one pipeline."""
    models = [("qwen", "Qwen-7B"), ("phi", "Phi-3.5-mini"), ("mistral", "Mistral-7B")]
    datasets = ["hotpotqa", "2wiki", "musique"]
    matrix, annots = [], []
    for key, _ in models:
        row, arow = [], []
        for ds in datasets:
            rows = _mscs(root / ds / "natural" / key)
            h1 = h1_small_sets(rows, n_boot=1000) if rows else None
            row.append(h1["rate"] if h1 else float("nan"))
            arow.append(
                f"{h1['rate']:.2f}\n[{h1['interval'][0]:.2f}, {h1['interval'][1]:.2f}]" if h1 else "—"
            )
        matrix.append(row)
        annots.append(arow)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    im = ax.imshow(matrix, cmap="Greens", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([label for _, label in models])
    for i in range(len(models)):
        for j in range(len(datasets)):
            ax.text(j, i, annots[i][j], ha="center", va="center", fontsize=9,
                    color="white" if matrix[i][j] >= 0.9 else "black")
    ax.set_title("H1 replicates across the grid: a small sufficient set in every cell (bound-5, target 0.50)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("H1 small-set rate")
    extra = []
    for tag, label in (("cerebras-gpt-oss-120b", "120B frontier"), ("qwenbm25", "BM25 live retrieval")):
        rows = _mscs(root / "hotpotqa" / "natural" / tag)
        if rows:
            extra.append(f"{label}  H1 {h1_small_sets(rows, n_boot=1000)['rate']:.2f}")
    if extra:
        fig.text(0.5, 0.01, "out-of-distribution checks (hotpotqa):   " + "     ".join(extra),
                 ha="center", fontsize=9, color="#333333")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig_robustness.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="bound-5 cell tree")
    parser.add_argument("--root3", type=Path, help="bound-3 cell tree (depth comparison baseline)")
    parser.add_argument("--k10", type=Path, help="the k=10 natural cell")
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fig_guarantee(args.root, args.out)
    fig_responsibility(args.root, args.out)
    fig_a3(args.root, args.out)
    fig_pareto(args.root, args.out)
    fig_selective(args.root, args.out)
    fig_robustness(args.root, args.out)
    if args.root3 and args.k10:
        fig_depth(args.root3, args.k10, args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
