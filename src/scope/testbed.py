"""Synthetic coalition games whose causal structure is known analytically.

Each game fixes a Boolean support structure over named passages and answers the subset query
directly — no model, no text — so the exact enumerator and the budgeted extractors are validated
against ground truth that is true by construction. OR-redundancy (any one supporter keeps the
answer) is the structure leave-one-out is blind to; AND-coalitions (all members needed) are its
dual; k-of-n interpolates between them.
"""
from __future__ import annotations

from typing import Callable, Iterable

from .game import Game


def formula_game(ids: Iterable[str], formula: Callable[[frozenset[str]], bool]) -> Game:
    return Game(ids, formula)


def or_game(supporters: Iterable[str], distractors: Iterable[str] = ()) -> Game:
    """Any one supporter alone reproduces the answer: every supporter is singly sufficient,
    none is singly necessary."""
    support = frozenset(supporters)
    return Game([*support, *distractors], lambda subset: bool(subset & support))


def and_game(members: Iterable[str], distractors: Iterable[str] = ()) -> Game:
    """Only the whole coalition reproduces the answer: every member is singly necessary,
    none is singly sufficient."""
    needed = frozenset(members)
    return Game([*needed, *distractors], lambda subset: needed <= subset)


def k_of_n_game(supporters: Iterable[str], k: int, distractors: Iterable[str] = ()) -> Game:
    """Any k of the supporters reproduce the answer."""
    support = frozenset(supporters)
    return Game([*support, *distractors], lambda subset: len(subset & support) >= k)


def parametric_game(ids: Iterable[str]) -> Game:
    """The answer survives any ablation — it comes from the model, not the context."""
    return Game(ids, lambda subset: True)
