"""Graded responsibility over an enumerated sufficient family, computed exactly.

The Chockler–Halpern degree of responsibility of a passage for the reproduced answer is
1/(1+k), where k is the fewest other passages whose removal makes the passage critical —
removing them keeps the answer, removing the passage too flips it. Under the family semantics
(the answer reproduces exactly when a member survives, the same support-determination reading
the guarantee uses), criticality has a combinatorial form: the removals must hit every member
that avoids the passage while sparing at least one member that contains it. Families are small,
so the minimal hitting set is exact brute force, not an approximation.

Two boundary cases carry the theory: a passage in every member is critical outright
(k=0, responsibility 1 — exactly the leave-one-out causal label of Theorem 1), and an OR-cover
of m singletons gives each carrier 1/m — the credit-dilution curve, derived rather than assumed.
"""
from __future__ import annotations

from itertools import combinations


def responsibility(passage: str, members: list[frozenset[str]]) -> float:
    """Degree of responsibility of one passage for the outcome the family carries."""
    members = [m for m in members if m]
    if not members or not any(passage in m for m in members):
        return 0.0
    avoiding = [m for m in members if passage not in m]
    if not avoiding:
        return 1.0
    containing = [m - {passage} for m in members if passage in m]
    pool = sorted(set().union(*avoiding) - {passage})
    for size in range(1, len(pool) + 1):
        for removal in combinations(pool, size):
            removed = set(removal)
            if all(m & removed for m in avoiding) and any(not (m & removed) for m in containing):
                return 1.0 / (1.0 + size)
    return 0.0


def case_responsibilities(members: list[frozenset[str]]) -> dict[str, float]:
    """Responsibility of every passage that appears in some member."""
    passages = sorted(set().union(*[m for m in members if m]) if any(members) else set())
    return {p: responsibility(p, members) for p in passages}


def max_responsibility(members: list[frozenset[str]]) -> float | None:
    """The strongest claim any single passage has on the error — 1.0 means a true culprit,
    <= 1/2 means responsibility is irreducibly shared. None when the family is empty."""
    values = case_responsibilities(members)
    return max(values.values()) if values else None
