from lineup.backends.base import Generation, LanguageModel, Scoring
from lineup.data.schema import Chunk, Recipe, Scenario

from scope import analyze
from scope.model_game import scenario_game


class FakeModel(LanguageModel):
    """Answers the planted wrong year when any wrong-value chunk is in the prompt."""

    def generate(self, messages, max_new_tokens=None):
        prompt = " ".join(message.content for message in messages)
        if "the premiere was in 1884" in prompt or "opened in 1884" in prompt:
            return Generation(text="The year was 1884, I believe.", token_ids=[], token_logprobs=[])
        return Generation(text="1902", token_ids=[], token_logprobs=[])

    def score(self, messages, response):
        return Scoring(tokens=[], token_ids=[], logprobs=[])


def make_scenario():
    chunks = [
        Chunk(chunk_id="g1", title="Playwright", text="The playwright was born in 1902.", provenance="gold"),
        Chunk(chunk_id="m1", title="Review", text="A review notes the premiere was in 1884.", provenance="misleading"),
        Chunk(chunk_id="w2", title="Archive", text="The archive says the theatre opened in 1884.", provenance="misleading"),
        Chunk(chunk_id="d1", title="Filler", text="An unrelated note about stagecraft.", provenance="distractor"),
    ]
    recipe = Recipe(
        seed=0,
        k=4,
        original_value="1902",
        intended_wrong_answer="1884",
        substitution_type="year",
        source_gold_chunk_id="g1",
        source_sentence_id=0,
        gold_chunk_ids=["g1"],
        misleading_chunk_id="m1",
        distractor_chunk_ids=["d1"],
        order=["g1", "m1", "w2", "d1"],
        decoy_chunk_id="w2",
    )
    return Scenario(qid="q1", question="When was the playwright born?", gold_answer="1902", chunks=chunks, recipe=recipe)


def test_redundant_wrong_support_is_loo_blind_end_to_end():
    game = scenario_game(FakeModel(), make_scenario(), target_answer="1884")
    structure = analyze(game)
    assert set(structure.minimal_sufficient) == {frozenset({"m1"}), frozenset({"w2"})}
    assert structure.minimal_necessary == (frozenset({"m1", "w2"}),)
    assert structure.loo_blind
    assert not structure.parametric


def test_answer_equivalence_uses_the_benchmark_comparator():
    # The fake answers verbosely ("The year was 1884, I believe.") while the target is the bare
    # value; answer_key treats them as one value, so a rephrase never counts as a flip.
    game = scenario_game(FakeModel(), make_scenario(), target_answer="1884")
    assert game.reproduces({"m1"})
    assert not game.reproduces({"g1", "d1"})
