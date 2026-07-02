"""Budgeted extraction of a sufficient set: three strategies over the one primitive.

``grow_prune`` works bottom-up along a priority order — cheapest when a ranking is trusted.
``shrink`` works top-down: ddmin-style block removal from the full context, discarding
low-priority blocks first; it is *anytime* — from the moment the full context verifies, it
always holds a sufficient set, so a budget cut degrades minimality, never sufficiency — and it
pays roughly log n when the responsible set is small. ``surrogate_beam`` lets the order-2
surrogate propose candidate sets for free and spends queries only verifying the best first — a
joint pair with a positive interaction term costs one verification, where bottom-up search
needs lookahead to reach it at all.

Every strategy returns a 1-minimal sufficient set on termination (no single member can be
dropped; minimal outright on monotone games) and counts each distinct model query against the
shared budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil
from typing import Iterable

from .game import Game
from .interactions import Effects
from .mscs import is_sufficient


@dataclass(frozen=True)
class Extraction:
    subset: frozenset[str] | None   # None: no sufficient set found within the budget
    queries: int                    # model queries this extraction spent

    @property
    def sufficient(self) -> bool:
        return self.subset is not None


def _prune(game: Game, current: frozenset[str], drop_order: Iterable[str], within_budget) -> frozenset[str]:
    for member in drop_order:
        if member not in current:
            continue
        if not within_budget():
            break
        without = current - {member}
        if is_sufficient(game, without):
            current = without
    return current


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

    current = _prune(game, current, list(reversed(ranked)), within_budget)
    return Extraction(subset=current, queries=spent())


def shrink(game: Game, order: Iterable[str] | None = None, budget: int | None = None) -> Extraction:
    """Minimize top-down: remove whole blocks of the context while the answer still reproduces,
    halving block size when stuck, down to single passages.

    ``order`` is the same most-responsible-first priority the other strategies take; blocks of
    low-priority passages are tried first, so a decent ranking steers what gets discarded. Once
    the full context verifies, every intermediate state is sufficient — running out of budget
    returns the current (possibly unminimized) set rather than nothing.
    """
    ranked = list(order) if order is not None else list(game.ids)
    start = game.queries

    def spent() -> int:
        return game.queries - start

    def within_budget() -> bool:
        return budget is None or spent() < budget

    if not within_budget():
        return Extraction(subset=None, queries=spent())
    if is_sufficient(game, frozenset()):
        return Extraction(subset=frozenset(), queries=spent())
    if not within_budget() or not is_sufficient(game, game.full):
        return Extraction(subset=None, queries=spent())

    current = list(reversed(ranked))            # least responsible first: discarded first
    granularity = 2
    while len(current) >= 2:
        size = ceil(len(current) / granularity)
        blocks = [current[i : i + size] for i in range(0, len(current), size)]
        reduced = False
        for block in blocks:
            if not within_budget():
                return Extraction(subset=frozenset(current), queries=spent())
            trial = [cid for cid in current if cid not in block]
            if is_sufficient(game, frozenset(trial)):
                current = trial
                granularity = max(granularity - 1, 2)
                reduced = True
                break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
    return Extraction(subset=frozenset(current), queries=spent())


def surrogate_beam(
    game: Game, effects: Effects, *, width: int = 8, max_size: int | None = None, budget: int | None = None
) -> Extraction:
    """Verify the surrogate's best guesses first: beam-search candidate sets on the fitted
    v(S) — pure arithmetic, no queries — then test them best-first and prune the winner.

    The candidate list always contains the empty set (the parametric guess) and every beam
    expansion, ordered by predicted value with smaller sets preferred on ties.
    """
    ids = list(effects.ids)

    def predicted(subset: frozenset[str]) -> float:
        value = effects.intercept + sum(effects.main.get(cid, 0.0) for cid in subset)
        value += sum(effects.pair(one, other) for one, other in combinations(sorted(subset), 2))
        return value

    limit = len(ids) if max_size is None else min(max_size, len(ids))
    visited = {frozenset(): predicted(frozenset())}
    beam = [frozenset()]
    for _ in range(limit):
        expansions = {subset | {cid} for subset in beam for cid in ids if cid not in subset}
        for subset in expansions:
            visited.setdefault(subset, predicted(subset))
        beam = sorted(expansions, key=lambda subset: (-round(visited[subset], 9), tuple(sorted(subset))))[:width]

    # Round before ranking so fit noise cannot shuffle genuinely tied predictions — the size
    # tie-break (test smaller sets first) is what keeps the query bill down.
    candidates = sorted(visited, key=lambda subset: (-round(visited[subset], 9), len(subset), tuple(sorted(subset))))
    drop_order = sorted(ids, key=lambda cid: effects.main.get(cid, 0.0))
    start = game.queries

    def spent() -> int:
        return game.queries - start

    def within_budget() -> bool:
        return budget is None or spent() < budget

    for candidate in candidates:
        if not within_budget():
            return Extraction(subset=None, queries=spent())
        if is_sufficient(game, candidate):
            return Extraction(subset=_prune(game, candidate, drop_order, within_budget), queries=spent())
    return Extraction(subset=None, queries=spent())
