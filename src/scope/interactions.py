"""Order-2 interaction effects over the subset game.

Fits the surrogate v(S) = b0 + sum bi xi + sum bij xi xj to the boolean reproduces-signal.
An additive (order-1) datamodel — ContextCite's shape — cannot represent redundancy at all;
with the pairwise terms it can: redundant support shows up as a *negative* bij (either member
suffices, together they add nothing new) and joint support as a *positive* bij (neither works
alone). Because the game is generation-only, this runs against API models that expose no
logprobs. Sampling and fitting go through the same cached Game, so the surrogate's cost lands
in the same query budget as everything else.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import chain, combinations

import numpy


@dataclass(frozen=True)
class Effects:
    ids: tuple[str, ...]
    intercept: float
    main: dict[str, float]
    pairwise: dict[frozenset, float]

    def pair(self, one: str, other: str) -> float:
        return self.pairwise.get(frozenset((one, other)), 0.0)


def _fit(ids: tuple[str, ...], masks: list[frozenset], values: list[float]) -> Effects:
    pairs = list(combinations(ids, 2))
    rows = [
        [1.0]
        + [1.0 if cid in mask else 0.0 for cid in ids]
        + [1.0 if one in mask and other in mask else 0.0 for one, other in pairs]
        for mask in masks
    ]
    solution = numpy.linalg.lstsq(numpy.array(rows), numpy.array(values, dtype=float), rcond=None)[0]
    main = dict(zip(ids, solution[1 : 1 + len(ids)]))
    pairwise = {frozenset(pair): value for pair, value in zip(pairs, solution[1 + len(ids) :])}
    return Effects(ids=ids, intercept=float(solution[0]), main={k: float(v) for k, v in main.items()}, pairwise={k: float(v) for k, v in pairwise.items()})


def exact_effects(game) -> Effects:
    """The surrogate fit over the full subset lattice — the reference the sampled fit converges
    to. Exponential in n, so guarded; meant for the testbed and small real cases."""
    if len(game.ids) > 12:
        raise ValueError(f"exact fit enumerates 2^{len(game.ids)} subsets; use sampled_effects")
    masks = [frozenset(combo) for combo in chain.from_iterable(combinations(game.ids, size) for size in range(len(game.ids) + 1))]
    values = [1.0 if game.reproduces(mask) else 0.0 for mask in masks]
    return _fit(game.ids, masks, values)


def sampled_effects(game, n_samples: int = 128, seed: int = 0) -> Effects:
    """The deployable fit: uniform random inclusion masks, plus the empty and full contexts as
    anchors. Repeated masks cost nothing (the game caches), so distinct queries never exceed
    min(n_samples, 2^n)."""
    rng = random.Random(seed)
    masks = [frozenset(), game.full]
    for _ in range(max(0, n_samples - 2)):
        masks.append(frozenset(cid for cid in game.ids if rng.random() < 0.5))
    values = [1.0 if game.reproduces(mask) else 0.0 for mask in masks]
    return _fit(game.ids, masks, values)


def interaction_order(effects: Effects) -> list[str]:
    """Rank chunks so synergy partners surface together.

    Greedy: each step scores a chunk by its main effect, plus its realized pairwise terms with
    everything already ranked, plus an optimistic half-share of its best remaining partnership —
    the lookahead is what lets a pure AND-pair (zero mains, positive pair) lead the order at all.
    Feed the result to grow_prune as ``order``.
    """
    remaining = list(effects.ids)
    ranked: list[str] = []
    while remaining:
        def score(cid: str) -> float:
            realized = sum(effects.pair(cid, other) for other in ranked)
            best = max((effects.pair(cid, other) for other in remaining if other != cid), default=0.0)
            return effects.main.get(cid, 0.0) + realized + max(0.0, best) / 2
        choice = max(remaining, key=score)
        ranked.append(choice)
        remaining.remove(choice)
    return ranked
