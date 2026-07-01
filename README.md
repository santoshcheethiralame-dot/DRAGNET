# SCoPE

Set-valued causal attribution for retrieval-augmented generation. When a RAG answer is wrong,
the responsible evidence is often a coalition of passages rather than one culprit — single-passage
attribution is then structurally incapable of naming it. SCoPE estimates the **minimal sufficient
causal set**: the smallest passage subset that alone reproduces the answer, extracted under an
explicit model-query budget, to be wrapped in a distribution-free coverage guarantee.

Builds on the [LINEUP benchmark](https://github.com/santoshcheethiralame-dot/LINEUP) for the
non-circular oracle, model backends, prompts, and data.

## Layout

- `src/scope/game.py` — the subset-query game (cached, query-counted): `reproduces(S)` is the one
  primitive everything else is defined over
- `src/scope/mscs.py` — exact minimal sufficient / necessary sets over the subset lattice; the
  ground truth extractors are validated against, never a deployable method
- `src/scope/extract.py` — budgeted extraction (grow along a priority order, then prune)
- `src/scope/testbed.py` — synthetic coalition games (OR / AND / k-of-n / arbitrary formula) whose
  causal structure is known analytically
- `src/scope/model_game.py` — the real game over a lineup scenario and backend

## Setup

```
pip install -e path/to/lineup
pip install -e .[dev]
pytest -q
```
