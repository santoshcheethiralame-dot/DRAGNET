"""Score the preregistered hypotheses from run artifacts, exactly as paper/prereg.md fixed them.

Each function takes parsed rows, applies the analysis plan committed before the data existed
(case bootstrap with percentile intervals, parametric cases outside coverage denominators,
violation cases excluded and counted), and returns the numbers plus a verdict against the
preregistered threshold. Nothing here touches a model; every input is a JSONL artifact a GPU
run already wrote.
"""
from __future__ import annotations

import random

from lineup.stats import percentile_interval

from .conformal import calibrate_depth, coverage_and_size


def _bootstrap_rate(flags: list[bool], n_boot: int, seed: int) -> tuple[float | None, float | None]:
    if not flags or not n_boot:
        return None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = [flags[rng.randrange(len(flags))] for _ in flags]
        draws.append(sum(sample) / len(sample))
    return percentile_interval(draws)


def h1_small_sets(mscs_rows: list[dict], *, n_boot: int = 1000, seed: int = 0) -> dict:
    """H1: among non-parametric wrong cases, the fraction with a minimal sufficient set within
    the enumerated bound. Preregistered pass: >= 0.50."""
    parametric = sum(1 for row in mscs_rows if row["parametric"])
    eligible = [row for row in mscs_rows if not row["parametric"]]
    flags = [bool(row["minimal_sufficient"]) for row in eligible]
    rate = sum(flags) / len(flags) if flags else None
    low, high = _bootstrap_rate(flags, n_boot, seed)
    return {
        "n": len(eligible),
        "parametric_fraction": parametric / len(mscs_rows) if mscs_rows else None,
        "rate": rate,
        "interval": (low, high),
        "passed": rate is not None and rate >= 0.5,
    }


def h2_coverage(items: list, *, alpha: float = 0.1, seeds: tuple[int, ...] = (0, 1, 2)) -> list[dict]:
    """H2: split-conformal prefix sets over the enumerated family, one seeded half split per
    seed, all reported. Preregistered pass per split: coverage within 2*SE of 1-alpha at the
    realized test n, mean size <= 4."""
    results = []
    for seed in seeds:
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        half = len(shuffled) // 2
        calibration, test = shuffled[:half], shuffled[half:]
        tau = calibrate_depth(calibration, alpha=alpha)
        if tau is None or not test:
            results.append({"seed": seed, "n_test": len(test), "passed": False})
            continue
        covered, size = coverage_and_size(test, tau)
        target = 1 - alpha
        slack = 2 * (target * (1 - target) / len(test)) ** 0.5
        results.append(
            {
                "seed": seed,
                "n_test": len(test),
                "tau": tau,
                "coverage": covered,
                "target": target,
                "slack": slack,
                "mean_size": size,
                "passed": covered >= target - slack and size <= 4,
            }
        )
    return results


def h3_designed_sufficient(validation_rows: list[dict], *, n_boot: int = 1000, seed: int = 0) -> dict:
    """H3a: the designed pair is behaviorally sufficient on the AND cell. Pass: >= 0.50."""
    flags = [bool(row["designed_sufficient"]) for row in validation_rows if row["structure"] == "and"]
    rate = sum(flags) / len(flags) if flags else None
    low, high = _bootstrap_rate(flags, n_boot, seed)
    return {
        "n": len(flags),
        "rate": rate,
        "interval": (low, high),
        "passed": rate is not None and rate >= 0.5,
    }


def h3_link_silent(role_cases: list, scenarios: dict) -> dict:
    """H3b: the link passage draws the oracle label *silent* at >= 2x the distractor rate,
    over the wrong cases of the AND cell."""
    link = {"silent": 0, "n": 0}
    distractor = {"silent": 0, "n": 0}
    for case in role_cases:
        if case.original_correct:
            continue
        scenario = scenarios.get(case.qid)
        if scenario is None:
            continue
        provenance = {chunk.chunk_id: chunk.provenance for chunk in scenario.chunks}
        for role in case.chunk_roles:
            bucket = {"link": link, "distractor": distractor}.get(provenance.get(role.chunk_id))
            if bucket is None:
                continue
            bucket["n"] += 1
            bucket["silent"] += int(role.role == "silent")
    link_rate = link["silent"] / link["n"] if link["n"] else None
    distractor_rate = distractor["silent"] / distractor["n"] if distractor["n"] else None
    passed = (
        link_rate is not None
        and distractor_rate is not None
        and link_rate >= 2 * distractor_rate
        and link_rate > 0
    )
    return {
        "link_rate": link_rate,
        "link_n": link["n"],
        "distractor_rate": distractor_rate,
        "distractor_n": distractor["n"],
        "passed": passed,
    }


def h4_paired_arms(
    extraction_rows: list[dict],
    *,
    arm_a: str = "beam",
    arm_b: str = "contextcite",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """H4: arm_a covers the designed set more often than arm_b, paired by case. One-sided
    paired case bootstrap; preregistered pass: p < 0.05."""
    covered: dict[str, dict[str, bool]] = {}
    for row in extraction_rows:
        if row["arm"] in (arm_a, arm_b):
            covered.setdefault(row["qid"], {})[row["arm"]] = bool(row["covered"])
    diffs = [
        int(arms[arm_a]) - int(arms[arm_b])
        for arms in covered.values()
        if arm_a in arms and arm_b in arms
    ]
    if not diffs:
        return {"n": 0, "passed": False}
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    at_or_below_zero = 0
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        if sum(sample) / len(sample) <= 0:
            at_or_below_zero += 1
    p = at_or_below_zero / n_boot
    return {
        "n": len(diffs),
        "rate_a": sum(1 for d in covered.values() if d.get(arm_a)) / len(diffs),
        "rate_b": sum(1 for d in covered.values() if d.get(arm_b)) / len(diffs),
        "mean_diff": mean,
        "p": p,
        "passed": mean > 0 and p < 0.05,
    }


def h5_loo_intersection(mscs_rows: list[dict], role_cases: list) -> dict:
    """H5: the leave-one-out causal set equals the intersection of the enumerated minimal
    sufficient family, after excluding recorded support-determination violations (their rate
    reported). Pass: >= 0.90. Cases with no set inside the bound cannot place the intersection
    and are excluded, counted."""
    loo = {
        case.qid: frozenset(role.chunk_id for role in case.chunk_roles if role.causal)
        for case in role_cases
        if not case.original_correct
    }
    violating = unevaluable = 0
    agreements = []
    for row in mscs_rows:
        if row["qid"] not in loo:
            continue
        if row["monotonicity_violations"]:
            violating += 1
            continue
        family = [frozenset(subset) for subset in row["minimal_sufficient"]]
        if not family:
            unevaluable += 1
            continue
        intersection = frozenset.intersection(*family)
        agreements.append(intersection == loo[row["qid"]])
    rate = sum(agreements) / len(agreements) if agreements else None
    return {
        "n": len(agreements),
        "violating": violating,
        "unevaluable": unevaluable,
        "rate": rate,
        "passed": rate is not None and rate >= 0.9,
    }
