"""The unified decision rule: singleton, set, or abstain — one calibrated mechanism.

The rule is group-conditional conformal with a size cap. Cases are binned by a label-free
confidence signal (a method's top1-top2 score margin — the benchmark's abstention signal); each
bin gets its own conformal depth tau from calibration. The verdict is then read off tau rather
than decided by separate machinery: tau = 1 names a singleton, 1 < tau <= cap returns the
depth-tau set, and tau beyond the cap (or infinite) abstains — the guarantee there would cost
more context than it explains, which is exactly when silence is the honest answer.

Both of the benchmark's remedies are corners of this rule: one bin with no cap is the plain
conformal set, and a cap of 1 is margin-gated selective attribution (top-1 or abstain). The
guarantee is group-conditional: within every answered bin, the returned set covers the target
family with probability >= 1 - alpha, provided the bin edges are chosen without labels (margin
quantiles are) and cases are exchangeable. Abstention is decided per bin, never per case, so it
cannot leak the test case's own difficulty into the promise.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable

from .conformal import DepthItem, calibrate_depth


@dataclass(frozen=True)
class Verdict:
    kind: str                       # "singleton" | "set" | "abstain"
    chunks: frozenset[str] | None   # None exactly when abstaining
    tau: float                      # the bin's calibrated depth (inf when uncalibratable)
    bin: int


@dataclass(frozen=True)
class DecisionRule:
    edges: tuple[float, ...]        # interior margin-quantile edges, label-free
    taus: tuple[float | None, ...]  # per-bin conformal depth; None for an empty bin
    alpha: float
    cap: int | None                 # abstain any bin whose tau exceeds this; None = no cap

    def _route(self, margin: float) -> tuple[int, float, bool]:
        index = bisect_right(self.edges, margin)
        tau = self.taus[index]
        if tau is None:
            return index, math.inf, False
        answered = tau != math.inf and (self.cap is None or tau <= self.cap)
        return index, tau, answered

    def decide(self, margin: float, order: Iterable[str]) -> Verdict:
        index, tau, answered = self._route(margin)
        if not answered:
            return Verdict(kind="abstain", chunks=None, tau=tau, bin=index)
        order = list(order)
        depth = min(int(tau), len(order))
        chunks = frozenset(order[:depth])
        return Verdict(kind="singleton" if depth == 1 else "set", chunks=chunks, tau=tau, bin=index)


def calibrate_rule(
    pairs: list[tuple[float, DepthItem]], alpha: float, *, bins: int = 4, cap: int | None = None
) -> DecisionRule:
    """Fit the rule on calibration cases: quantile-bin the margins, then run the same conformal
    order statistic within each bin. Small bins inflate their own tau (the (n+1)(1-alpha) index
    saturates), so an over-binned rule degrades toward abstention, never toward overclaiming."""
    margins = sorted(margin for margin, _ in pairs)
    edges = tuple(margins[len(margins) * i // bins] for i in range(1, bins)) if margins else ()
    grouped: list[list[DepthItem]] = [[] for _ in range(bins)]
    for margin, item in pairs:
        grouped[bisect_right(edges, margin)].append(item)
    taus = tuple(calibrate_depth(group, alpha) for group in grouped)
    return DecisionRule(edges=edges, taus=taus, alpha=alpha, cap=cap)


def calibrate_stratified_rules(
    pairs_by_stratum: dict[str, list[tuple[float, DepthItem]]],
    alpha: float,
    *,
    bins: int = 4,
    cap: int | None = None,
) -> dict[str, DecisionRule]:
    """Mondrian calibration: one rule per stratum (model, dataset, whatever the deployment
    conditions on), so the guarantee holds *conditionally* on each stratum rather than only on
    the pooled mixture — the failure mode a marginal tau hides is one stratum quietly paying
    for another's coverage. Small strata inherit the conservative small-bin behavior: they
    drift toward abstention, never toward overclaiming."""
    return {
        stratum: calibrate_rule(pairs, alpha, bins=bins, cap=cap)
        for stratum, pairs in pairs_by_stratum.items()
    }


def evaluate_stratified(
    rules: dict[str, DecisionRule],
    pairs_by_stratum: dict[str, list[tuple[float, DepthItem]]],
) -> dict:
    """Per-stratum reports plus the pooled aggregate (weighted by stratum size)."""
    per_stratum = {
        stratum: evaluate_rule(rules[stratum], pairs)
        for stratum, pairs in pairs_by_stratum.items()
        if stratum in rules
    }
    n = sum(report["n"] for report in per_stratum.values())
    answered = sum(
        round((1 - report["abstain_rate"]) * report["n"]) for report in per_stratum.values() if report["n"]
    )
    covered = sum(
        report["answered_coverage"] * round((1 - report["abstain_rate"]) * report["n"])
        for report in per_stratum.values()
        if report["n"] and report["answered_coverage"] is not None
    )
    return {
        "per_stratum": per_stratum,
        "pooled": {
            "n": n,
            "abstain_rate": (n - answered) / n if n else None,
            "answered_coverage": covered / answered if answered else None,
        },
    }


def evaluate_rule(rule: DecisionRule, pairs: list[tuple[float, DepthItem]]) -> dict:
    """Test-split report: verdict shares, coverage among answered cases (the guarantee), and the
    context cost paid for it, overall and per bin."""
    bins = [
        {"n": 0, "answered": 0, "covered": 0, "size": 0.0, "tau": tau}
        for tau in rule.taus
    ]
    for margin, item in pairs:
        index, tau, answered = rule._route(margin)
        bucket = bins[index]
        bucket["n"] += 1
        if not answered:
            continue
        bucket["answered"] += 1
        bucket["covered"] += int(item.depth is not None and item.depth <= tau)
        bucket["size"] += min(tau, item.n_chunks)

    n = sum(bucket["n"] for bucket in bins)
    answered = sum(bucket["answered"] for bucket in bins)
    covered = sum(bucket["covered"] for bucket in bins)
    size = sum(bucket["size"] for bucket in bins)
    singleton = sum(bucket["answered"] for bucket in bins if bucket["tau"] == 1)
    return {
        "n": n,
        "abstain_rate": (n - answered) / n if n else None,
        "singleton_rate": singleton / n if n else None,
        "answered_coverage": covered / answered if answered else None,
        "mean_answered_size": size / answered if answered else None,
        "bins": bins,
    }
