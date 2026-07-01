from scope import analyze, is_necessary, is_sufficient, minimal_necessary_sets, minimal_sufficient_sets
from scope.testbed import and_game, formula_game, k_of_n_game, or_game, parametric_game


def test_or_redundancy_is_loo_blind():
    game = or_game(["s1", "s2"], distractors=["d1", "d2"])
    structure = analyze(game)
    assert set(structure.minimal_sufficient) == {frozenset({"s1"}), frozenset({"s2"})}
    assert structure.minimal_necessary == (frozenset({"s1", "s2"}),)
    assert structure.singleton_sufficient == {"s1", "s2"}
    assert structure.singleton_necessary == frozenset()
    assert structure.loo_blind


def test_and_coalition_is_the_dual():
    game = and_game(["a", "b"], distractors=["d1"])
    structure = analyze(game)
    assert structure.minimal_sufficient == (frozenset({"a", "b"}),)
    assert set(structure.minimal_necessary) == {frozenset({"a"}), frozenset({"b"})}
    assert structure.singleton_necessary == {"a", "b"}
    assert not structure.loo_blind


def test_k_of_n_interpolates():
    game = k_of_n_game(["s1", "s2", "s3", "s4"], k=2, distractors=["d1"])
    sufficient = minimal_sufficient_sets(game)
    necessary = minimal_necessary_sets(game)
    assert all(len(s) == 2 and s <= {"s1", "s2", "s3", "s4"} for s in sufficient)
    assert len(sufficient) == 6
    assert all(len(s) == 3 for s in necessary)
    assert len(necessary) == 4


def test_mixed_formula():
    ids = ["a", "b", "c"]
    game = formula_game(ids, lambda s: ("a" in s and "b" in s) or "c" in s)
    assert set(minimal_sufficient_sets(game)) == {frozenset({"c"}), frozenset({"a", "b"})}
    assert set(minimal_necessary_sets(game)) == {frozenset({"a", "c"}), frozenset({"b", "c"})}


def test_parametric_error_is_the_empty_set():
    structure = analyze(parametric_game(["a", "b"]))
    assert structure.parametric
    assert structure.minimal_sufficient == (frozenset(),)
    assert structure.minimal_necessary == ()


def test_max_size_bounds_the_search():
    game = and_game(["a", "b"], distractors=["d1"])
    assert minimal_sufficient_sets(game, max_size=1) == []
    assert minimal_sufficient_sets(game, max_size=2) == [frozenset({"a", "b"})]


def test_sufficiency_and_necessity_agree_with_definitions():
    game = or_game(["s1", "s2"])
    assert is_sufficient(game, {"s1"})
    assert not is_necessary(game, {"s1"})
    assert is_necessary(game, {"s1", "s2"})


def test_queries_are_cached():
    game = or_game(["s1", "s2"], distractors=["d1"])
    game.reproduces({"s1"})
    game.reproduces({"s1"})
    assert game.queries == 1
