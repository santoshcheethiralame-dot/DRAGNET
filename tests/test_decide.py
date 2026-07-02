import math
import random

from scope.conformal import DepthItem, calibrate_depth
from scope.decide import (
    DecisionRule,
    calibrate_rule,
    calibrate_stratified_rules,
    evaluate_rule,
    evaluate_stratified,
)


def _item(depth, n_chunks=8, key="k"):
    return DepthItem(key=key, depth=depth, n_chunks=n_chunks)


def test_one_bin_no_cap_is_the_plain_conformal_set():
    # Reduction 1: the stage-3 conformal set is the bins=1, cap=None corner of the rule.
    pairs = [(random.Random(i).random(), _item(1 + i % 4)) for i in range(40)]
    rule = calibrate_rule(pairs, alpha=0.2, bins=1)
    assert rule.taus == (calibrate_depth([item for _, item in pairs], alpha=0.2),)
    verdict = rule.decide(0.5, [f"c{i}" for i in range(8)])
    assert verdict.chunks == frozenset(f"c{i}" for i in range(int(rule.taus[0])))


def test_cap_one_is_selective_attribution():
    # Reduction 2: cap=1 answers with a single pick or stays silent — the margin-gated
    # top-1-or-abstain remedy.
    rng = random.Random(0)
    pairs = [
        (margin, _item(1 if margin > 0.5 else rng.randint(2, 8)))
        for margin in (rng.random() for _ in range(200))
    ]
    rule = calibrate_rule(pairs, alpha=0.2, bins=4, cap=1)
    kinds = {rule.decide(margin, [f"c{i}" for i in range(8)]).kind for margin, _ in pairs}
    assert kinds <= {"singleton", "abstain"}
    assert "singleton" in kinds and "abstain" in kinds


def test_verdicts_read_off_tau():
    rule = DecisionRule(edges=(0.3, 0.6), taus=(None, 3.0, 1.0), alpha=0.1, cap=4)
    order = ["a", "b", "c", "d", "e"]
    assert rule.decide(0.1, order).kind == "abstain"
    middle = rule.decide(0.5, order)
    assert middle.kind == "set" and middle.chunks == frozenset({"a", "b", "c"})
    confident = rule.decide(0.9, order)
    assert confident.kind == "singleton" and confident.chunks == frozenset({"a"})


def test_the_cap_turns_an_expensive_bin_into_abstention():
    pairs = [(0.5, _item(6)) for _ in range(30)]
    capped = calibrate_rule(pairs, alpha=0.1, bins=1, cap=3)
    assert capped.decide(0.5, [f"c{i}" for i in range(8)]).kind == "abstain"
    uncapped = calibrate_rule(pairs, alpha=0.1, bins=1)
    assert uncapped.decide(0.5, [f"c{i}" for i in range(8)]).kind == "set"


def test_uncoverable_mass_pushes_a_bin_to_abstain_not_overclaim():
    pairs = [(0.5, _item(None)) for _ in range(20)] + [(0.5, _item(1)) for _ in range(10)]
    rule = calibrate_rule(pairs, alpha=0.1, bins=1)
    assert rule.taus[0] == math.inf
    assert rule.decide(0.5, ["a", "b"]).kind == "abstain"


def test_group_conditional_coverage_holds_and_size_adapts():
    rng = random.Random(0)
    pairs = []
    for _ in range(800):
        margin = rng.random()
        depth = 1 if rng.random() < 0.5 + 0.45 * margin else rng.randint(2, 8)
        pairs.append((margin, _item(depth)))
    calibration, test = pairs[0::2], pairs[1::2]
    rule = calibrate_rule(calibration, alpha=0.2, bins=4)
    report = evaluate_rule(rule, test)

    assert report["answered_coverage"] >= 0.75          # target 0.80, finite-sample slack
    for bucket in report["bins"]:
        if bucket["answered"]:
            assert bucket["covered"] / bucket["answered"] >= 0.70
    # Confident cases pay less context than murky ones — the point of the rule.
    assert rule.taus[-1] < rule.taus[0]
    assert report["singleton_rate"] > 0


def test_empty_bins_abstain():
    rule = DecisionRule(edges=(0.5,), taus=(None, 2.0), alpha=0.1, cap=None)
    assert rule.decide(0.2, ["a", "b", "c"]).kind == "abstain"
    assert rule.decide(0.8, ["a", "b", "c"]).kind == "set"


def _stratified_fixture():
    # An easy stratum (depth 1) and a hard one (depth 5). The hard stratum is kept *below*
    # alpha of the mixture — exactly the regime where the pooled quantile never reaches its
    # tail, so the marginal guarantee is honest while the stratum silently fails.
    rng = random.Random(0)
    easy = [(rng.random(), _item(1)) for _ in range(600)]
    hard = [(rng.random(), _item(5)) for _ in range(60)]
    return easy, hard


def test_pooled_calibration_undercovers_the_hard_stratum():
    easy, hard = _stratified_fixture()
    pooled = calibrate_rule(easy[:300] + hard[:30], alpha=0.1, bins=1)
    marginal = evaluate_rule(pooled, easy[300:] + hard[30:])
    hard_only = evaluate_rule(pooled, hard[30:])
    assert marginal["answered_coverage"] >= 0.9   # the marginal guarantee holds...
    assert hard_only["answered_coverage"] == 0.0  # ...while the hard stratum silently pays for it


def test_stratified_calibration_restores_conditional_coverage():
    easy, hard = _stratified_fixture()
    rules = calibrate_stratified_rules(
        {"easy": easy[:300], "hard": hard[:30]}, alpha=0.1, bins=1
    )
    report = evaluate_stratified(rules, {"easy": easy[300:], "hard": hard[30:]})
    assert report["per_stratum"]["hard"]["answered_coverage"] >= 0.9
    assert report["per_stratum"]["easy"]["answered_coverage"] >= 0.9
    assert report["pooled"]["answered_coverage"] >= 0.9
    # The hard stratum pays with a bigger set, not with silent under-coverage.
    assert rules["hard"].taus[0] > rules["easy"].taus[0]


def test_single_stratum_reduces_to_the_plain_rule():
    easy, _ = _stratified_fixture()
    rules = calibrate_stratified_rules({"all": easy[:300]}, alpha=0.2, bins=2)
    assert rules["all"] == calibrate_rule(easy[:300], alpha=0.2, bins=2)


def test_an_empty_stratum_abstains_everywhere():
    rules = calibrate_stratified_rules({"empty": []}, alpha=0.1, bins=2)
    assert rules["empty"].decide(0.5, ["a", "b"]).kind == "abstain"
