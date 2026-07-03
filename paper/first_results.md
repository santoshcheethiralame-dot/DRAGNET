# First artifacts — scored 2026-07-03

Cells on disk: hotpotqa natural {qwen, phi} (mscs.jsonl); hotpotqa AND+OR {qwen, phi, mistral};
musique AND+OR qwen; 2wiki AND+OR qwen (pipelines only). mistral and musique carry per-case
validation/extraction rows; natural cells for mistral/musique/2wiki and orders.jsonl are still
to run. All numbers below are CPU recomputes from the artifacts via score_prereg / run_symmetry /
run_decision_rule; bootstrap = 1000 unless noted.

## Preregistered verdicts

- **H1 PASS.** Small-set rate qwen 0.93 [0.87, 0.99] (n=67, parametric 0.11), phi 0.90
  [0.82, 0.97] (n=67, parametric 0.06). Min-size histograms qwen {1:52, 2:7, 3:3},
  phi {1:37, 2:14, 3:9}.
- **H2 mixed — the coverage–size law, on the ContextCite order.** Pooled 134 items:
  alpha=0.2 → tau=3–4, coverage 0.78–0.88 at size 3–4 (passes); alpha=0.1 → tau=inf
  (whole context). The interaction order (orders.jsonl) is the untested lever.
- **H3a borderline.** designed-sufficient: qwen 0.61, mistral 0.50 [0.33, 0.67],
  musique/qwen 0.51 [0.37, 0.66], phi 0.27. exact = 0.00 everywhere.
- **H3b FAIL in all five cells.** link-silent vs distractor-silent: qwen 0.26/0.26,
  phi 0.41/0.42, mistral 0.50/0.52, musique 0.34/0.30, 2wiki 0.40/0.27. The designed silent
  link is not detectably more silent than a distractor.
- **H4 FAIL on the designed target.** beam vs contextcite designed-coverage: mistral 0.20 vs
  0.25 (p 0.74), musique 0.05 vs 0.05 (p 1.0). See the reframe below.
- **H5 FAIL + finding.** LOO = intersection on clean cases only 0.78 (qwen, n=18) / 0.59
  (phi, n=17); the A3 violation rate dominates (below).

## Exploratory (not preregistered — labelled as such)

- **Functional ambiguity 0.52** (70/134 non-parametric organic cases have >1 minimal
  sufficient set) — the Theorem-5 precondition measured. Lexical near-duplicates are rare
  (case rate 0.03–0.11 across thresholds 0.6–0.4; touching a member 0.01–0.06): the ambiguity
  is functional, not lexical — retrieval dedup cannot remove it.
- **Behavioral-coverage reframe of H4.** Joining extraction.jsonl against the enumerated
  families in validation.jsonl: every arm covers the *behavioral* truth at 0.93 (mistral,
  n=15) / 1.00 (musique, n=18) while designed coverage sits at 0.06–0.33, and sufficient-rate
  is 1.00 everywhere. Extraction finds real sufficient sets reliably; the *planted* sets are
  simply not what the model uses. H3/H4 read as a verdict on the chain-split instrument, not
  on extraction. Arms do not differentiate on behavioral coverage at this n.
- **A3 non-monotonicity is severe, not marginal.** Violating cases: qwen 52/75, phi 47/71;
  median violating pairs 9; qwen has 43/75 cases at >=5 violations. 29 cases per model are
  both violating and ambiguous. Adding a passage frequently flips a reproduced answer —
  context interference is the norm in these games, which reshapes Thm 1 to its approximate
  form and independently motivates minimal sets (more context is not monotonically better).
- **Unified decision rule on the mscs family** (pooled natural, contextcite margin+ranking):
  alpha=0.2, one bin → tau=4, answered coverage 0.88 at size 4.0; alpha=0.1 → full abstention
  (the rule degrades to silence rather than overclaim, as designed). Bins>1 saturate at this
  n. Re-run with the interaction order and margin when orders.jsonl lands.

## Depth check — k=10 natural cell (qwen/hotpotqa, scored 2026-07-03)

The retrieval-depth run replicates every k=6 number: H1 0.93 [0.85, 0.98] (n=54, parametric
0.10), min-size histogram {1:37, 2:7, 3:6}, functional ambiguity 0.52, A3-violating 44/60.
**Minimal sufficient set size is flat in k** — the exact escape Prop 4 predicts, against the
necessary-set size that scaled ~0.45k in the benchmark. Conformal on the ContextCite order at
alpha=0.2: tau=3 of 10 passages (coverage 0.85–0.89) — the absolute set budget did not grow
with the context, so the guarantee is relatively tighter at depth. alpha=0.1 stays vacuous on
this order (the interaction-order test is pending). H5 again fails under the violation rate
(0.42 on n=12 clean cases).

## What remains to run

Natural cells (pipeline + mscs + orders) for mistral/hotpotqa, qwen/musique, qwen/2wiki;
orders.jsonl backfill on the two existing hotpotqa natural cells. The designed arm needs no
further GPU.
