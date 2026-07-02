"""Approximate case symmetry via near-duplicate passages — the Theorem 5 rate.

Identifiability fails when a case admits a relabeling that swaps interchangeable passages while
fixing every observable. Designed OR cells realize the symmetry exactly (decoys are constructed
near-copies); organic cells realize it approximately wherever retrieval returned near-duplicate
passages. The functions here measure that: pairwise token overlap between a case's passages,
the pairs above a threshold, and how often such a pair touches the responsible candidates —
the frequency of the regime where single-set attribution is non-identifiable. The similarity
value itself is the observable gap the theorem's approximation error is stated in.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")


def token_set(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of the word sets — dependency-free and symmetric."""
    tokens_a, tokens_b = token_set(a), token_set(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def near_duplicate_pairs(chunks: list, threshold: float) -> list[tuple[str, str, float]]:
    """Passage pairs at or above the threshold, most similar first. ``chunks`` carry
    ``chunk_id`` and ``text``."""
    pairs = []
    for i, first in enumerate(chunks):
        for second in chunks[i + 1 :]:
            value = similarity(first.text, second.text)
            if value >= threshold:
                pairs.append((first.chunk_id, second.chunk_id, value))
    return sorted(pairs, key=lambda pair: pair[2], reverse=True)


def case_symmetry(chunks: list, members: frozenset[str], threshold: float) -> dict:
    """One case's symmetry read: does any near-duplicate pair exist, does one touch the
    responsible candidates (that is the pair that moves a family member), and how close is the
    closest pair regardless of threshold."""
    pairs = near_duplicate_pairs(chunks, threshold)
    touching = [pair for pair in pairs if pair[0] in members or pair[1] in members]
    closest = max(
        (similarity(a.text, b.text) for i, a in enumerate(chunks) for b in chunks[i + 1 :]),
        default=0.0,
    )
    return {
        "has_pair": bool(pairs),
        "touches_members": bool(touching),
        "n_pairs": len(pairs),
        "closest": closest,
        "pairs": pairs,
    }


def symmetry_rates(reads: list[dict]) -> dict:
    """Pooled rates over per-case reads: the approximate-symmetry frequency, and the sharper
    rate where the pair touches the responsible candidates."""
    n = len(reads)
    if not n:
        return {"n": 0, "pair_rate": None, "member_rate": None}
    return {
        "n": n,
        "pair_rate": sum(read["has_pair"] for read in reads) / n,
        "member_rate": sum(read["touches_members"] for read in reads) / n,
    }
