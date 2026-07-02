"""Shared loading for the CPU runners: one record per wrong case of a cell.

Families come in four modes. ``designed`` targets the planted carriers (designed.jsonl, else
retrofitted from the recipe); ``behavioral`` targets the responsible sets the model actually
exhibits — leave-one-out causal singletons from roles.jsonl plus the jointly-necessary pairs the
leave-two-out probe recorded in coalition_proof.jsonl; ``fixer`` targets repair — the passages
whose lone removal restores the correct answer, the set a debugger actually wants; ``mscs``
targets the enumerated minimal sufficient sets from mscs.jsonl (the run_natural_mscs artifact) —
the exact object the guarantee is about. An empty family means the case is uncoverable and is
kept, not dropped; cases a mode has no record for are skipped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lineup.data.coalition import from_recipe, read_designed
from lineup.data.serialization import read_generations, read_predictions, read_roles, read_scenarios

from scope.designed import designed_family, necessary_family


@dataclass
class CaseData:
    key: str
    qid: str
    source: str              # the cell path: .../dataset/condition/model
    scenario: object         # the lineup Scenario, for building the model game
    model_answer: str        # the wrong answer whose support is in question
    presented: list          # passage order as shown to the model
    ranking: list | None     # contextcite score order; None when no prediction exists
    margin: float            # contextcite top1 - top2 score, the confidence signal
    family: tuple            # target sufficient/necessary sets; empty = uncoverable


def _synergy_pairs(cell: Path) -> dict:
    path = cell / "coalition_proof.jsonl"
    if not path.exists():
        return {}
    pairs: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("has_synergy") and record.get("synergy_pair"):
            pairs.setdefault(record["qid"], []).append(record["synergy_pair"])
    return pairs


def load_cases(cell: Path, family_mode: str) -> list[CaseData]:
    scenarios = {s.qid: s for s in read_scenarios(cell / "scenarios.jsonl")}
    wrong = [g for g in read_generations(cell / "generations.jsonl") if not g.is_correct]

    if family_mode == "designed":
        designed_path = cell / "designed.jsonl"
        if designed_path.exists():
            designed = {d.qid: d for d in read_designed(designed_path)}
        else:
            designed = {qid: d for qid, s in scenarios.items() if (d := from_recipe(s)) is not None}
        families = {qid: designed_family(d.cover_chunk_ids, d.threshold) for qid, d in designed.items()}
    elif family_mode == "fixer":
        families = {
            case.qid: tuple(frozenset({r.chunk_id}) for r in case.chunk_roles if r.now_correct)
            for case in read_roles(cell / "roles.jsonl")
            if not case.original_correct
        }
    elif family_mode == "mscs":
        families = {}
        for line in (cell / "mscs.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["parametric"]:
                # The empty set is sufficient, so any prefix would count as covered — parametric
                # cases sit outside the guarantee's denominator (their fraction is reported by
                # the prereg scorer, not silently dropped here and forgotten).
                continue
            families[record["qid"]] = tuple(frozenset(subset) for subset in record["minimal_sufficient"])
    else:
        causal = {
            case.qid: [role.chunk_id for role in case.chunk_roles if role.causal]
            for case in read_roles(cell / "roles.jsonl")
            if not case.original_correct
        }
        synergy = _synergy_pairs(cell)
        families = {qid: necessary_family(ids, synergy.get(qid, ())) for qid, ids in causal.items()}

    rankings, margins = {}, {}
    for prediction in read_predictions(cell / "predictions.jsonl"):
        if prediction.method == "contextcite":
            ranked = sorted(prediction.chunk_scores, key=lambda s: s.score, reverse=True)
            rankings[prediction.qid] = [score.chunk_id for score in ranked]
            margins[prediction.qid] = (
                ranked[0].score - ranked[1].score if len(ranked) >= 2 else 0.0
            )

    cases = []
    for generation in wrong:
        scenario, family = scenarios.get(generation.qid), families.get(generation.qid)
        if scenario is None or family is None:
            continue
        cases.append(
            CaseData(
                key=f"{cell}/{generation.qid}",
                qid=generation.qid,
                source=str(cell),
                scenario=scenario,
                model_answer=generation.model_answer,
                presented=[chunk.chunk_id for chunk in scenario.chunks],
                ranking=rankings.get(generation.qid),
                margin=margins.get(generation.qid, 0.0),
                family=family,
            )
        )
    return cases
