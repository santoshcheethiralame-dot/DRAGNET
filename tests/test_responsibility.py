from scope.responsibility import case_responsibilities, max_responsibility, responsibility
from scope.spectrum import hierarchy_holds, moebius_from_family, unsupported_terms


def f(*sets):
    return [frozenset(s) for s in sets]


def test_sole_culprit_bears_full_responsibility():
    members = f({"a"})
    assert responsibility("a", members) == 1.0
    assert responsibility("b", members) == 0.0


def test_and_pair_both_critical():
    members = f({"a", "b"})
    assert responsibility("a", members) == 1.0
    assert responsibility("b", members) == 1.0


def test_or_cover_dilutes_to_one_over_m():
    two = f({"a"}, {"b"})
    assert responsibility("a", two) == 0.5
    three = f({"a"}, {"b"}, {"c"})
    assert responsibility("a", three) == 1.0 / 3.0
    assert max_responsibility(three) == 1.0 / 3.0


def test_intersection_member_recovers_loo_causality():
    # b sits in every member -> critical with no removals, exactly the LOO-causal label.
    members = f({"a", "b"}, {"b", "c"})
    values = case_responsibilities(members)
    assert values["b"] == 1.0
    assert values["a"] == 0.5 and values["c"] == 0.5


def test_empty_family_has_no_responsibility():
    assert max_responsibility([]) is None
    assert max_responsibility([frozenset()]) is None


def test_moebius_or_pair_is_hierarchical():
    coefficients = moebius_from_family(f({"a"}, {"b"}))
    assert coefficients[frozenset({"a"})] == 1
    assert coefficients[frozenset({"b"})] == 1
    assert coefficients[frozenset({"a", "b"})] == -1
    assert hierarchy_holds(f({"a"}, {"b"}))


def test_moebius_and_pair_is_pure_synergy():
    coefficients = moebius_from_family(f({"a", "b"}))
    assert coefficients == {frozenset({"a", "b"}): 1}
    assert unsupported_terms(coefficients) == [frozenset({"a", "b"})]
    assert not hierarchy_holds(f({"a", "b"}))


def test_mixed_family_hierarchy():
    # a alone suffices, and the pair {b, c} suffices only jointly: the pair term is
    # unsupported even though the family also has a singleton elsewhere.
    members = f({"a"}, {"b", "c"})
    assert not hierarchy_holds(members)
    assert unsupported_terms(moebius_from_family(members)) == [frozenset({"b", "c"})]
