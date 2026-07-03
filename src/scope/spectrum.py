"""Exact Moebius spectrum of a sufficient family — the object interaction methods estimate.

The family's indicator (does this subset contain a member?) is a monotone boolean function;
its Moebius transform is the exact interaction spectrum scalable attribution methods
approximate from samples. Enumerated families make it computable in closed form, so the
structural assumptions those methods rest on become testable rather than assumed. The one that
matters: hierarchy — that a nonzero higher-order term is accompanied by nonzero lower-order
subsets. An AND-pair violates it exactly (pure synergy, zero mains); an OR-cover satisfies it.
"""
from __future__ import annotations

from itertools import combinations


def moebius_from_family(members: list[frozenset[str]]) -> dict[frozenset[str], int]:
    """Moebius coefficients of the family indicator over the union of its members.

    Exact zeta-inversion over the lattice of the member union — passages outside every member
    have zero coefficients and are omitted. Guarded to 16 elements (2^16 subsets)."""
    members = [m for m in members if m]
    if not members:
        return {}
    universe = sorted(set().union(*members))
    if len(universe) > 16:
        raise ValueError(f"universe too large for exact spectrum ({len(universe)} passages)")
    coefficients: dict[frozenset[str], int] = {}
    for size in range(0, len(universe) + 1):
        for combo in combinations(universe, size):
            subset = frozenset(combo)
            value = int(any(m <= subset for m in members))
            total = sum(coefficients[t] for t in coefficients if t < subset)
            coefficient = value - total
            if coefficient:
                coefficients[subset] = coefficient
    return coefficients


def unsupported_terms(coefficients: dict[frozenset[str], int]) -> list[frozenset[str]]:
    """Nonzero interaction terms of order >= 2 whose proper nonempty subsets all carry zero —
    the pure-synergy terms a hierarchy-assuming method has no lower-order trail toward."""
    nonzero = {s for s, c in coefficients.items() if c and s}
    return sorted(
        (
            s
            for s in nonzero
            if len(s) >= 2 and not any(t for t in nonzero if t < s)
        ),
        key=sorted,
    )


def hierarchy_holds(members: list[frozenset[str]]) -> bool:
    """True when every interaction term has lower-order support — the assumption under
    hierarchical/sparse interaction attribution, checked exactly."""
    return not unsupported_terms(moebius_from_family(members))
