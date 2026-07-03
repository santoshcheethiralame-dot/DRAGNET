# DRAGNET

**D**istribution-free **RAG** **N**onconformity **E**rror **T**raceback — set-valued causal attribution for retrieval-augmented generation. When a RAG answer is wrong,
the responsible evidence is often a coalition of passages rather than one culprit — a lineup
cannot name a single suspect, so you cast a dragnet: a calibrated set guaranteed to contain the
responsible passages, as small as the evidence allows. DRAGNET estimates the **minimal
sufficient causal set** — the smallest passage subset that alone reproduces the answer — under
an explicit model-query budget, wrapped in a distribution-free coverage guarantee.

Builds on the [LINEUP benchmark](https://github.com/santoshcheethiralame-dot/LINEUP) for the
non-circular oracle, model backends, prompts, and data.

## Layout

- `src/dragnet/game.py` — the subset-query game (cached, query-counted): `reproduces(S)` is the one
  primitive everything else is defined over
- `src/dragnet/mscs.py` — exact minimal sufficient / necessary sets over the subset lattice; the
  ground truth extractors are validated against, never a deployable method
- `src/dragnet/extract.py` — budgeted extraction (grow along a priority order, then prune)
- `src/dragnet/interactions.py` — order-2 subset surrogate: redundancy = negative pairwise terms,
  joint support = positive; `interaction_order` feeds the extractor
- `src/dragnet/designed.py` — designed-truth metrics: the set-coverage predicate and the
  designed-vs-behavioral comparison
- `src/dragnet/conformal.py` — split-conformal attribution sets: prefix-depth nonconformity over a
  sufficient-set family; the culprit-rank score is the size-1 corner
- `src/dragnet/testbed.py` — synthetic coalition games (OR / AND / k-of-n / arbitrary formula) whose
  causal structure is known analytically
- `src/dragnet/model_game.py` — the real game over a lineup scenario and backend

## Setup

```
pip install -e path/to/lineup
pip install -e .[dev]
pytest -q
```
