"""Minimal sufficient and necessary causal sets, computed exactly.

A subset S is *sufficient* if the model on S alone reproduces the answer value, and *necessary*
if removing S from the full context flips it. A minimal such set has no proper subset with the
same property. Exact enumeration walks the subset lattice bottom-up, so it is exponential in
general; over the small retrieval contexts of the benchmark (and bounded set sizes) it serves as
the ground truth budgeted extractors are validated against — never as a deployable method.

The bridge to the benchmark's leave-one-out oracle: a chunk is LOO-causal exactly when its
singleton is a necessary set. OR-redundancy is the structure leave-one-out is blind to — every
supporter is singly sufficient, none is singly necessary — and AND-coalitions are its dual.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from .game import Game


def is_sufficient(game: Game, subset: Iterable[str]) -> bool:
    return game.reproduces(frozenset(subset))


def is_necessary(game: Game, subset: Iterable[str]) -> bool:
    return not game.reproduces(game.full - frozenset(subset))


def _minimal_sets(game: Game, holds, max_size: int | None) -> list[frozenset[str]]:
    # Bottom-up over the lattice: a candidate containing an already-found minimal set is skipped,
    # so by induction everything appended has no proper subset with the property — minimal by
    # construction, with no monotonicity assumption on the model.
    limit = len(game.ids) if max_size is None else min(max_size, len(game.ids))
    found: list[frozenset[str]] = []
    for size in range(0, limit + 1):
        for combo in combinations(game.ids, size):
            candidate = frozenset(combo)
            if any(smaller <= candidate for smaller in found):
                continue
            if holds(game, candidate):
                found.append(candidate)
    return found


def minimal_sufficient_sets(game: Game, max_size: int | None = None) -> list[frozenset[str]]:
    """All minimal sufficient sets of size <= max_size, smallest first.

    The empty set counts: if the model reproduces the answer with no context at all, the error
    is parametric and no passage set explains it.
    """
    return _minimal_sets(game, is_sufficient, max_size)


def minimal_necessary_sets(game: Game, max_size: int | None = None) -> list[frozenset[str]]:
    """All minimal necessary sets of size <= max_size: removing the set flips the answer value,
    removing any proper subset does not."""
    return _minimal_sets(game, is_necessary, max_size)


@dataclass(frozen=True)
class CausalStructure:
    """The exact causal structure of one case, in the benchmark's vocabulary."""

    minimal_sufficient: tuple[frozenset[str], ...]
    minimal_necessary: tuple[frozenset[str], ...]

    @property
    def parametric(self) -> bool:
        """The empty context already reproduces the answer — nothing in the context explains it."""
        return frozenset() in self.minimal_sufficient

    @property
    def singleton_sufficient(self) -> frozenset[str]:
        """Chunks that alone reproduce the answer (over-determination when there are several)."""
        return frozenset(x for s in self.minimal_sufficient if len(s) == 1 for x in s)

    @property
    def singleton_necessary(self) -> frozenset[str]:
        """Chunks whose lone removal flips the answer — exactly the leave-one-out causal label."""
        return frozenset(x for s in self.minimal_necessary if len(s) == 1 for x in s)

    @property
    def loo_blind(self) -> bool:
        """A necessary set exists but no singleton one: the redundant coalition leave-one-out
        cannot see."""
        return bool(self.minimal_necessary) and not self.singleton_necessary


def analyze(game: Game, max_size: int | None = None) -> CausalStructure:
    return CausalStructure(
        minimal_sufficient=tuple(minimal_sufficient_sets(game, max_size)),
        minimal_necessary=tuple(minimal_necessary_sets(game, max_size)),
    )
