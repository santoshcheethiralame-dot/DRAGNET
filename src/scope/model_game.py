"""The real game: a lineup scenario queried through a lineup backend.

Answer equivalence reuses the benchmark's ``answer_key``, so "S reproduces the answer" here means
exactly what "removing the chunk changes the answer" means to the leave-one-out oracle — the two
never drift. Kept in its own module so the core stays import-light; everything lineup lives here.
"""
from __future__ import annotations

from .game import Game


def scenario_game(model, scenario, target_answer: str) -> Game:
    """Build the subset game for one (scenario, model, answer) instance.

    ``target_answer`` is the answer whose causal support is in question — for error attribution,
    the model's wrong answer on the full context. Chunks keep their presented order inside every
    subset, so position effects stay fixed while membership varies.
    """
    from lineup.oracle import answer_key
    from lineup.prompt import build_messages_for

    gold = scenario.gold_answer
    wrong = scenario.recipe.intended_wrong_answer
    target = answer_key(target_answer, gold, wrong)

    def query(subset: frozenset[str]) -> bool:
        shown = [chunk for chunk in scenario.chunks if chunk.chunk_id in subset]
        answer = model.generate(build_messages_for(scenario.question, shown)).text.strip()
        return answer_key(answer, gold, wrong) == target

    return Game([chunk.chunk_id for chunk in scenario.chunks], query)
