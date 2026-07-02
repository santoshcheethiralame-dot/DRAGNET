from types import SimpleNamespace

from scope.conformal import depth_item
from scope.prereg import (
    h1_small_sets,
    h2_coverage,
    h3_designed_sufficient,
    h3_link_silent,
    h4_paired_arms,
    h5_loo_intersection,
)


def mscs_row(qid, minimal_sufficient, parametric=False, violations=0):
    return {
        "qid": qid,
        "parametric": parametric,
        "minimal_sufficient": minimal_sufficient,
        "monotonicity_violations": violations,
    }


def role_case(qid, causal_ids, roles=(), correct=False):
    chunk_roles = [SimpleNamespace(chunk_id=cid, causal=cid in causal_ids, role=role) for cid, role in roles] or [
        SimpleNamespace(chunk_id=cid, causal=True, role="culprit") for cid in causal_ids
    ]
    return SimpleNamespace(qid=qid, original_correct=correct, chunk_roles=chunk_roles)


def test_h1_excludes_parametric_from_the_denominator():
    rows = [
        mscs_row("a", [["x"]]),
        mscs_row("b", []),
        mscs_row("c", [["y", "z"]]),
        mscs_row("d", [[]], parametric=True),
    ]
    result = h1_small_sets(rows, n_boot=200)
    assert result["n"] == 3
    assert result["parametric_fraction"] == 0.25
    assert abs(result["rate"] - 2 / 3) < 1e-9
    assert result["passed"]


def test_h1_fails_below_half():
    rows = [mscs_row("a", [])] * 3 + [mscs_row("b", [["x"]])]
    assert not h1_small_sets(rows, n_boot=0)["passed"]


def test_h2_covers_when_the_family_leads_the_order():
    items = [depth_item(str(i), ["a", "b", "c", "d"], (frozenset({"a"}),)) for i in range(40)]
    splits = h2_coverage(items, alpha=0.1, seeds=(0, 1, 2))
    assert len(splits) == 3
    for split in splits:
        assert split["passed"]
        assert split["coverage"] == 1.0
        assert split["mean_size"] <= 4


def test_h2_fails_when_nothing_is_covered():
    items = [depth_item(str(i), ["a", "b"], (frozenset({"missing"}),)) for i in range(20)]
    for split in h2_coverage(items, alpha=0.1, seeds=(0,)):
        assert not split["passed"]


def test_h3a_rate_over_and_rows_only():
    rows = [
        {"structure": "and", "designed_sufficient": True},
        {"structure": "and", "designed_sufficient": True},
        {"structure": "and", "designed_sufficient": False},
        {"structure": "or", "designed_sufficient": False},
    ]
    result = h3_designed_sufficient(rows, n_boot=200)
    assert result["n"] == 3
    assert abs(result["rate"] - 2 / 3) < 1e-9
    assert result["passed"]


def test_h3b_link_silent_against_distractors():
    scenarios = {
        "q": SimpleNamespace(
            chunks=[
                SimpleNamespace(chunk_id="l", provenance="link"),
                SimpleNamespace(chunk_id="d1", provenance="distractor"),
                SimpleNamespace(chunk_id="d2", provenance="distractor"),
            ]
        )
    }
    cases = [role_case("q", causal_ids={"l"}, roles=[("l", "silent"), ("d1", "inert"), ("d2", "inert")])]
    result = h3_link_silent(cases, scenarios)
    assert result["link_rate"] == 1.0
    assert result["distractor_rate"] == 0.0
    assert result["passed"]


def test_h4_paired_bootstrap_detects_the_better_arm():
    rows = []
    for i in range(30):
        rows.append({"qid": str(i), "arm": "beam", "covered": i < 24})
        rows.append({"qid": str(i), "arm": "contextcite", "covered": i < 8})
    result = h4_paired_arms(rows, n_boot=2000)
    assert result["n"] == 30
    assert result["mean_diff"] > 0
    assert result["p"] < 0.05
    assert result["passed"]


def test_h4_no_signal_no_pass():
    rows = []
    for i in range(30):
        rows.append({"qid": str(i), "arm": "beam", "covered": i % 2 == 0})
        rows.append({"qid": str(i), "arm": "contextcite", "covered": i % 2 == 0})
    assert not h4_paired_arms(rows, n_boot=500)["passed"]


def test_h5_intersection_matches_loo_on_and_and_or():
    rows = [
        mscs_row("and", [["a", "b"]]),
        mscs_row("or", [["a"], ["b"]]),
        mscs_row("violating", [["a"]], violations=2),
        mscs_row("unbounded", []),
    ]
    cases = [
        role_case("and", causal_ids={"a", "b"}),
        role_case("or", causal_ids=set(), roles=[("a", "misleading"), ("b", "misleading")]),
        role_case("violating", causal_ids={"a"}),
        role_case("unbounded", causal_ids=set(), roles=[("a", "inert")]),
    ]
    result = h5_loo_intersection(rows, cases)
    assert result["n"] == 2
    assert result["violating"] == 1
    assert result["unevaluable"] == 1
    assert result["rate"] == 1.0
    assert result["passed"]


def test_h5_disagreement_fails():
    rows = [mscs_row(str(i), [["a"]]) for i in range(10)]
    cases = [role_case(str(i), causal_ids={"b"}, roles=[("a", "inert"), ("b", "culprit")]) for i in range(10)]
    result = h5_loo_intersection(rows, cases)
    assert result["rate"] == 0.0
    assert not result["passed"]
