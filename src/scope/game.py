from __future__ import annotations

from typing import Callable, Iterable


class Game:
    """The subset-query game behind set-valued attribution.

    ``reproduces(S)`` asks: shown exactly the passages in S, does the model still produce the
    same answer value it gave on the full context? Sufficiency, necessity, and minimal causal
    sets are all defined over this single primitive. Queries are cached by subset and counted,
    so every algorithm on top pays — and reports — its true model-call budget.
    """

    def __init__(self, ids: Iterable[str], query: Callable[[frozenset[str]], bool]):
        self.ids = tuple(ids)
        self._query = query
        self._cache: dict[frozenset[str], bool] = {}

    @property
    def full(self) -> frozenset[str]:
        return frozenset(self.ids)

    @property
    def queries(self) -> int:
        """Distinct subsets actually sent to the model (cache hits are free)."""
        return len(self._cache)

    def reproduces(self, subset: Iterable[str]) -> bool:
        subset = frozenset(subset)
        if subset not in self._cache:
            self._cache[subset] = bool(self._query(subset))
        return self._cache[subset]
