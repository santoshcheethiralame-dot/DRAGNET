"""Budgeted extraction of a sufficient set: grow along a priority order, then prune.

The grow pass adds passages in the given order until the set reproduces the answer; the prune
pass tries dropping each member, keeping only what sufficiency still needs. On monotone games the
result is a minimal sufficient set; in general it is 1-minimal — no single member can be dropped —
which is the honest guarantee a black-box model admits. Cost is at most 2n+1 queries, against the
exponential lattice the exact enumerator walks.

The ``order`` parameter is the seam for smarter extraction: pass chunks ranked by an attribution
method's scores and the grow pass reaches sufficiency early, shrinking both the budget and the
pruning work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .game import Game
from .mscs import is_sufficient


@dataclass(frozen=True)
class Extraction:
    subset: frozenset[str] | None   # None: no sufficient set found within the budget
    queries: int                    # model queries this extraction spent

    @property
    def sufficient(self) -> bool:
        return self.subset is not None


def grow_prune(game: Game, order: Iterable[str] | None = None, budget: int | None = None) -> Extraction:
    """Return a 1-minimal sufficient set, or None if the budget runs out first.

    Grow tests prefixes of ``order`` (the empty prefix first — the parametric check), so the
    first sufficient prefix costs at most n+1 queries; prune then drops members from the lowest
    priority up, at one query each.
    """
    ranked = list(order) if order is not None else list(game.ids)
    start = game.queries

    def spent() -> int:
        return game.queries - start

    def within_budget() -> bool:
        return budget is None or spent() < budget

    current: frozenset[str] | None = None
    for size in range(0, len(ranked) + 1):
        if not within_budget():
            return Extraction(subset=None, queries=spent())
        prefix = frozenset(ranked[:size])
        if is_sufficient(game, prefix):
            current = prefix
            break
    if current is None:
        return Extraction(subset=None, queries=spent())

    for member in reversed(ranked):
        if member not in current:
            continue
        if not within_budget():
            break
        without = current - {member}
        if is_sufficient(game, without):
            current = without
    return Extraction(subset=current, queries=spent())
