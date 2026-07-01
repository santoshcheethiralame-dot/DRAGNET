"""Score sets against the coalition a generator planted, and against live behavior.

Designed truth says which passages were built to carry the wrong answer and how many of them
jointly suffice; behavioral truth is whatever the exact enumerator finds on the model. The
construction is a hypothesis about behavior — the agreement rate between the two is a measured
quantity, never an assumption.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

from .mscs import CausalStructure


def designed_family(cover: Iterable[str], threshold: int) -> tuple[frozenset[str], ...]:
    """The sufficient sets a construction intends: every threshold-sized subset of the cover."""
    return tuple(frozenset(combo) for combo in combinations(tuple(cover), threshold))


def set_covers(candidate: Iterable[str] | None, family: Iterable[frozenset[str]]) -> bool:
    """The set-coverage predicate: does the returned set contain a sufficient set?"""
    if candidate is None:
        return False
    candidate = frozenset(candidate)
    return any(member <= candidate for member in family)


def compare(structure: CausalStructure, family: Iterable[frozenset[str]]) -> dict:
    """Designed-vs-behavioral verdict for one case.

    ``exact``: the model's minimal sufficient sets are precisely the designed ones.
    ``designed_sufficient``: every designed set at least works (some behavioral minimal set sits
    inside it) — the weaker, construction-validating read.
    ``parametric``: the empty context already reproduces the answer, so no construction explains it.
    """
    family = tuple(family)
    behavioral = set(structure.minimal_sufficient)
    return {
        "exact": behavioral == set(family),
        "designed_sufficient": bool(behavioral) and all(
            any(member <= designed for member in behavioral) for designed in family
        ),
        "parametric": structure.parametric,
    }
