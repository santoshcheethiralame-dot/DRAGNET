from scope import analyze, grow_prune
from scope.testbed import and_game, formula_game, or_game, parametric_game


def test_prune_recovers_a_supporter_from_a_bad_order():
    game = or_game(["s1", "s2"], distractors=["d1", "d2"])
    result = grow_prune(game, order=["d1", "d2", "s1", "s2"])
    assert result.subset == frozenset({"s1"})
    assert result.queries <= 2 * len(game.ids) + 1


def test_and_coalition_is_returned_whole():
    game = and_game(["a", "b"], distractors=["d1"])
    result = grow_prune(game, order=["d1", "a", "b"])
    assert result.subset == frozenset({"a", "b"})


def test_budget_exhaustion_returns_none():
    game = or_game(["s1"], distractors=["d1", "d2"])
    result = grow_prune(game, order=["d1", "d2", "s1"], budget=3)
    assert result.subset is None
    assert result.queries == 3


def test_result_is_one_minimal_not_globally_minimal():
    # Sufficient exactly on {a, b} and on {c}: grown from the front, {a, b} survives pruning
    # (neither singleton works), even though {c} is smaller. The 1-minimality boundary.
    game = formula_game(["a", "b", "c"], lambda s: s == frozenset({"a", "b"}) or s == frozenset({"c"}))
    result = grow_prune(game, order=["a", "b", "c"])
    assert result.subset == frozenset({"a", "b"})
    assert set(analyze(game).minimal_sufficient) == {frozenset({"c"}), frozenset({"a", "b"})}


def test_parametric_answer_costs_one_query():
    result = grow_prune(parametric_game(["a", "b"]))
    assert result.subset == frozenset()
    assert result.queries == 1


def test_good_order_spends_less():
    game = or_game(["s1", "s2"], distractors=["d1", "d2"])
    informed = grow_prune(game, order=["s1", "s2", "d1", "d2"])
    assert informed.subset == frozenset({"s1"})
    blind = grow_prune(or_game(["s1", "s2"], distractors=["d1", "d2"]), order=["d1", "d2", "s1", "s2"])
    assert informed.queries < blind.queries
