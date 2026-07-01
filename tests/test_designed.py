from lineup.backends.base import Generation, LanguageModel, Scoring
from lineup.data.coalition import ChainSplitScenarioBuilder, from_recipe
from lineup.data.scenario import ScenarioBuilder
from lineup.data.schema import Chunk, QAExample

from scope import analyze
from scope.designed import compare, designed_family, set_covers
from scope.model_game import scenario_game

AND_POOL = {"year": [], "number": [], "entity": ["Antwerp", "Rotterdam"]}
OR_POOL = {"year": [], "number": [], "entity": ["Alexandre Bartholdi", "Henri Banks"]}


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


def _or_example() -> QAExample:
    gold_a = Chunk(
        "q::0", "Eiffel Tower",
        "The Eiffel Tower was designed by Gustave Eiffel.",
        ["The Eiffel Tower was designed by Gustave Eiffel."],
        provenance="gold", supporting_sentence_ids=(0,),
    )
    gold_b = Chunk(
        "q::1", "Paris", "Paris is in France.", ["Paris is in France."],
        provenance="gold", supporting_sentence_ids=(0,),
    )
    distractors = [
        Chunk(f"q::{i}", f"D{i}", f"Topic {i} text.", [f"Topic {i} text."])
        for i in range(2, 8)
    ]
    return QAExample("q", "Who designed the Eiffel Tower?", "Gustave Eiffel", [gold_a, gold_b], distractors)


class RuleModel(LanguageModel):
    """Answers wrong exactly when every trigger string is in the prompt (AND), or when any
    member of a trigger group is (OR by passing one-element groups)."""

    def __init__(self, trigger_groups, wrong: str, gold: str):
        self.trigger_groups = trigger_groups
        self.wrong = wrong
        self.gold = gold

    def generate(self, messages, max_new_tokens=None):
        prompt = " ".join(message.content for message in messages)
        fired = all(any(trigger in prompt for trigger in group) for group in self.trigger_groups)
        return Generation(text=self.wrong if fired else self.gold, token_ids=[], token_logprobs=[])

    def score(self, messages, response):
        return Scoring(tokens=[], token_ids=[], logprobs=[])


def test_set_covers_is_the_coverage_predicate():
    family = designed_family(["a", "b", "c"], threshold=2)
    assert set_covers({"a", "b"}, family)
    assert set_covers({"a", "b", "d"}, family)
    assert not set_covers({"a", "d"}, family)
    assert not set_covers(None, family)


def test_and_construction_behaves_as_designed_end_to_end():
    scenario, designed = ChainSplitScenarioBuilder(answer_pool=AND_POOL, k=6, seed=0).build(_bridge_example())
    link = next(chunk for chunk in scenario.chunks if chunk.provenance == "link")
    value = next(chunk for chunk in scenario.chunks if chunk.provenance == "misleading")
    wrong = scenario.recipe.intended_wrong_answer

    answer = f"The answer is {wrong}."
    model = RuleModel([[link.text], [value.text]], wrong=answer, gold="Ghent")
    game = scenario_game(model, scenario, target_answer=answer)
    structure = analyze(game, max_size=3)

    family = designed_family(designed.cover_chunk_ids, designed.threshold)
    assert structure.minimal_sufficient == family
    verdict = compare(structure, family)
    assert verdict["exact"] and verdict["designed_sufficient"] and not verdict["parametric"]
    # The AND pair is exactly what leave-one-out CAN see: both members singly necessary.
    assert structure.singleton_necessary == frozenset(designed.cover_chunk_ids)


def test_or_construction_behaves_as_designed_end_to_end():
    scenario = ScenarioBuilder(answer_pool=OR_POOL, k=7, seed=0, n_decoys=1).build(_or_example())
    designed = from_recipe(scenario)
    wrong = scenario.recipe.intended_wrong_answer

    # Any passage carrying the wrong value sustains the wrong answer: OR over the cover.
    model = RuleModel([[wrong]], wrong=wrong, gold="Gustave Eiffel")
    game = scenario_game(model, scenario, target_answer=wrong)
    structure = analyze(game, max_size=3)

    family = designed_family(designed.cover_chunk_ids, designed.threshold)
    assert set(structure.minimal_sufficient) == set(family)
    assert compare(structure, family)["exact"]
    # The redundant cover is invisible to leave-one-out: nothing is singly necessary.
    assert structure.loo_blind
