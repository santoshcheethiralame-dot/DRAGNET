from scope import analyze, grow_prune
from scope.extract import shrink, surrogate_beam
from scope.interactions import exact_effects, sampled_effects
from scope.mscs import is_sufficient
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


def test_shrink_finds_a_minimal_set_top_down():
    result = shrink(or_game(["s1", "s2"], distractors=["d1", "d2"]))
    assert result.subset in ({"s1"}, {"s2"})
    pair = shrink(and_game(["a", "b"], distractors=["d1", "d2"]))
    assert pair.subset == frozenset({"a", "b"})


def test_shrink_discards_the_low_priority_supporter():
    game = or_game(["s1", "s2"], distractors=["d1", "d2"])
    result = shrink(game, order=["s2", "s1", "d1", "d2"])
    assert result.subset == frozenset({"s2"})


def test_shrink_is_anytime_under_a_budget():
    game = and_game(["a", "b"], distractors=["d1", "d2", "d3", "d4"])
    result = shrink(game, budget=4)
    # Too few queries to finish minimizing, but what it holds is already sufficient.
    assert result.sufficient
    assert is_sufficient(game, result.subset)


def test_shrink_handles_the_edges():
    assert shrink(parametric_game(["a", "b"])).subset == frozenset()
    never = formula_game(["a", "b"], lambda s: False)
    assert shrink(never).subset is None


def test_shrink_pays_log_like_costs_on_a_wide_context():
    supporters = ["s1"]
    distractors = [f"d{i}" for i in range(15)]
    result = shrink(or_game(supporters, distractors=distractors))
    assert result.subset == frozenset({"s1"})
    blind_grow = grow_prune(
        or_game(supporters, distractors=distractors), order=[*distractors, "s1"]
    )
    assert result.queries < blind_grow.queries


def test_beam_verifies_the_coalition_in_one_shot():
    game = and_game(["a", "b"], distractors=["d1", "d2", "d3"])
    result = surrogate_beam(game, exact_effects(and_game(["a", "b"], distractors=["d1", "d2", "d3"])))
    assert result.subset == frozenset({"a", "b"})
    assert result.queries <= 3   # one verification, at most two prune probes


def test_beam_prefers_the_smaller_of_tied_candidates():
    game = or_game(["s1", "s2"], distractors=["d1"])
    result = surrogate_beam(game, exact_effects(or_game(["s1", "s2"], distractors=["d1"])))
    assert result.subset in ({"s1"}, {"s2"})
    assert result.queries <= 2


def test_beam_finds_the_parametric_answer_first():
    game = parametric_game(["a", "b", "c"])
    effects = sampled_effects(parametric_game(["a", "b", "c"]), n_samples=32, seed=0)
    result = surrogate_beam(game, effects)
    assert result.subset == frozenset()
    assert result.queries == 1


def test_beam_respects_the_budget():
    game = and_game(["a", "b"], distractors=["d1"])
    result = surrogate_beam(game, exact_effects(and_game(["a", "b"], distractors=["d1"])), budget=0)
    assert result.subset is None and result.queries == 0


def test_beam_on_a_mixed_structure_with_sampled_effects():
    def build():
        return formula_game(
            ["a", "b", "c", "d1", "d2", "d3"],
            lambda s: ("a" in s and "b" in s) or "c" in s,
        )

    effects = sampled_effects(build(), n_samples=128, seed=0)
    game = build()
    result = surrogate_beam(game, effects)
    assert result.subset in (frozenset({"c"}), frozenset({"a", "b"}))
