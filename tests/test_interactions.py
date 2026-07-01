from pytest import approx

from lineup.backends.base import Generation, LanguageModel, Scoring
from lineup.data.coalition import ChainSplitScenarioBuilder
from lineup.data.schema import Chunk, QAExample

from scope import analyze, grow_prune
from scope.interactions import exact_effects, interaction_order, sampled_effects
from scope.model_game import scenario_game
from scope.testbed import and_game, formula_game, or_game


def test_exact_effects_split_redundancy_from_synergy():
    redundant = exact_effects(or_game(["s1", "s2"], distractors=["d1"]))
    # v = x1 + x2 - x1*x2 exactly: redundancy is a negative pairwise term.
    assert redundant.main["s1"] == approx(1.0)
    assert redundant.main["s2"] == approx(1.0)
    assert redundant.pair("s1", "s2") == approx(-1.0)
    assert redundant.main["d1"] == approx(0.0, abs=1e-9)

    joint = exact_effects(and_game(["a", "b"], distractors=["d1"]))
    # v = xa*xb exactly: joint support is a positive pairwise term and no main effect.
    assert joint.pair("a", "b") == approx(1.0)
    assert joint.main["a"] == approx(0.0, abs=1e-9)
    assert joint.pair("a", "d1") == approx(0.0, abs=1e-9)


def test_sampled_fit_recovers_the_exact_coefficients():
    exact = exact_effects(or_game(["s1", "s2"], distractors=["d1", "d2"]))
    sampled = sampled_effects(or_game(["s1", "s2"], distractors=["d1", "d2"]), n_samples=200, seed=0)
    assert sampled.pair("s1", "s2") == approx(exact.pair("s1", "s2"), abs=1e-6)
    assert sampled.main["s1"] == approx(exact.main["s1"], abs=1e-6)


def test_sampled_effects_are_deterministic_and_cache_bounded():
    game = or_game(["s1", "s2"], distractors=["d1", "d2"])
    first = sampled_effects(game, n_samples=100, seed=7)
    assert game.queries <= 16  # distinct masks never exceed the 2^4 lattice
    second = sampled_effects(or_game(["s1", "s2"], distractors=["d1", "d2"]), n_samples=100, seed=7)
    assert first == second


def test_interaction_order_surfaces_the_and_pair():
    game = and_game(["a", "b"], distractors=["d1", "d2", "d3"])
    order = interaction_order(exact_effects(game))
    assert set(order[:2]) == {"a", "b"}
    result = grow_prune(game, order=order)
    assert result.subset == frozenset({"a", "b"})
    assert result.queries <= 5  # empty, {a}, {a,b}, then one prune probe each


def test_interaction_order_leads_with_a_supporter_under_redundancy():
    game = or_game(["s1", "s2"], distractors=["d1", "d2"])
    order = interaction_order(exact_effects(game))
    assert order[0] in {"s1", "s2"}
    result = grow_prune(game, order=order)
    assert result.subset in ({"s1"}, {"s2"})
    assert result.queries <= 3


def test_guided_extraction_on_a_mixed_structure():
    def build():
        return formula_game(
            ["a", "b", "c", "d1", "d2", "d3"],
            lambda s: ("a" in s and "b" in s) or "c" in s,
        )

    game = build()
    effects = sampled_effects(game, n_samples=128, seed=0)
    guided = grow_prune(game, order=interaction_order(effects))
    assert guided.subset in (frozenset({"c"}), frozenset({"a", "b"}))
    assert set(analyze(build()).minimal_sufficient) == {frozenset({"c"}), frozenset({"a", "b"})}


class _PairModel(LanguageModel):
    """Wrong exactly when both planted passages are in the prompt."""

    def __init__(self, triggers, wrong):
        self.triggers = triggers
        self.wrong = wrong

    def generate(self, messages, max_new_tokens=None):
        prompt = " ".join(message.content for message in messages)
        fired = all(trigger in prompt for trigger in self.triggers)
        return Generation(text=self.wrong if fired else "Ghent", token_ids=[], token_logprobs=[])

    def score(self, messages, response):
        return Scoring(tokens=[], token_ids=[], logprobs=[])


def _bridge_example() -> QAExample:
    link = Chunk(
        "q::0", "Landmark Tower",
        "The Landmark Tower was designed by Otto Vane.",
        ["The Landmark Tower was designed by Otto Vane."],
        provenance="gold", supporting_sentence_ids=(0,),
    )
    value = Chunk(
        "q::1", "Otto Vane",
        "Otto Vane was born in Ghent.",
        ["Otto Vane was born in Ghent."],
        provenance="gold", supporting_sentence_ids=(0,),
    )
    distractors = [
        Chunk(f"q::{i}", f"Topic {chr(63 + i)} Hall", f"Notes on topic {i}.", [f"Notes on topic {i}."])
        for i in range(2, 8)
    ]
    return QAExample(
        "q", "Where was the designer of the Landmark Tower born?", "Ghent", [link, value], distractors
    )


def test_planted_and_pair_carries_the_positive_interaction():
    pool = {"year": [], "number": [], "entity": ["Antwerp", "Rotterdam"]}
    scenario, designed = ChainSplitScenarioBuilder(answer_pool=pool, k=6, seed=0).build(_bridge_example())
    link = next(chunk for chunk in scenario.chunks if chunk.provenance == "link")
    value = next(chunk for chunk in scenario.chunks if chunk.provenance == "misleading")
    wrong = scenario.recipe.intended_wrong_answer

    model = _PairModel([link.text, value.text], wrong=wrong)
    game = scenario_game(model, scenario, target_answer=wrong)
    effects = sampled_effects(game, n_samples=128, seed=0)

    planted_pair = frozenset({link.chunk_id, value.chunk_id})
    assert max(effects.pairwise, key=effects.pairwise.get) == planted_pair
    assert effects.pairwise[planted_pair] > 0.5

    order = interaction_order(effects)
    assert set(order[:2]) == planted_pair
    result = grow_prune(game, order=order)
    assert result.subset == planted_pair
