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

## Full natural grid — scored 2026-07-03 (all three orders per cell)

**H1 passes in every cell of the grid.** Small-set rate: qwen/hotpotqa 0.93, phi/hotpotqa 0.90,
qwen k=10 0.93, **mistral/hotpotqa 0.72 [0.60, 0.83]**, qwen/musique 0.92 [0.85, 0.97],
qwen/2wiki 0.95 [0.90, 0.99]. Mistral is the honest low end: 28% of its errors have no
sufficient set within size 3 — its problem is reach, not ambiguity.

**Ambiguity is stable ~0.5 for qwen across datasets** (hotpotqa 0.52, k=10 0.52, musique 0.54,
2wiki 0.52); mistral 0.38. **A3 violations pervasive everywhere**: mistral 0.48, qwen/hotpotqa
0.69, musique 0.70, 2wiki 0.80.

**H2, pooled over the three orders-bearing cells (227 items):**

- alpha=0.1 — **tau=inf for every order.** The coverage ceiling is the family's reach:
  27/227 (11.9%) non-parametric cases have no set within the size-3 bound, capping coverage
  at 0.88 < 0.90. At this enumeration bound the 90% guarantee is unreachable by ANY ranking —
  Prop 4 as measured fact, driven by mistral (15/53 no-set).
- alpha=0.2 — the coalition-aware orders buy set size: interaction/shapley tau=3 (cov
  0.81–0.89) vs contextcite tau=4. A 25% smaller set at equal coverage; shapley and
  interaction are comparable (shapley slightly steadier per-cell: 2wiki tau=3, musique tau=4).

So the ranking helps at the margin; the binding constraint is enumeration depth. The direct
lever: re-enumerate mscs at max_size 5 (63 vs 42 queries/case at k=6) to lift the ceiling
toward ~0.95 and retest alpha=0.1.

**H5 fails in all new cells too** (0.40 / 0.45 / 0.56) with violation exclusions dominating
(28 / 70 / 80 cases) — Thm 1 holds only in its approximate form; report the violation rate as
a first-class quantity.

## Family structure — graded responsibility and the exact spectrum (2026-07-03, exploratory)

Computed exactly from the enumerated families (all six natural cells, pooled n=372 evaluable
wrong cases; parametric and out-of-bound cases excluded and counted).

**Graded responsibility (Chockler–Halpern over the family semantics).** In **37%** of cases no
passage bears responsibility above 1/2 (per-cell 0.26–0.42); 63% have a true culprit
(responsibility 1.0 = a member-intersection passage = the LOO-causal label, Thm 1's object).
The distribution is exactly {1, 1/2, 1/3, ...} by arithmetic — 1/(1+k) for integer k — so
"nothing in (1/2, 1)" is bookkeeping, not a finding; the substantive quantity is the 37%
irreducibly-shared rate. **That rate independently reproduces the benchmark's pooled
no-single-culprit rate (37%) from a different instrument on different cells** — the LOO role
matrix and the CH-responsibility computation over enumerated families converge on the same
number. Credit dilution is now derived, not assumed: OR-covers of m carriers each get 1/m.

**Hierarchy of the interaction spectrum.** The exact Moebius spectrum of the family indicator
violates the hierarchy assumption (nonzero high-order terms accompanied by nonzero lower-order
subsets) in **51%** of cases pooled (0.34–0.63 per cell; 560 pure-synergy terms). Scalable
interaction-attribution methods lean on precisely this assumption — and it fails on the
AND-coalition cases where interaction attribution matters most. Caveat: computed on the
monotone family indicator at the enumeration bound, i.e. on the object those methods try to
recover, not the raw (non-monotone) game.

## The alpha=0.1 retest at bound-5 — scored 2026-07-03 (deep-MSCS runs)

Re-enumeration at max_size 5 lifted the family reach exactly as Prop 4 predicts: pooled
no-set-in-bound fell 27/227 → 20/361 (5.5%), ceiling 0.88 → ~0.94. H1 strengthens everywhere
(qwen/hotpotqa 0.96, phi 0.97, musique 0.97, 2wiki 0.99, mistral 0.79).

**The 90% guarantee at small sets is now achieved where the theory says it can be:** the
shapley order passes all three seeds on musique (tau=4, coverage 0.92–0.94) and 2wiki
(tau=3–4, coverage 0.93–1.00); qwen/hotpotqa passes 2/3 on the contextcite order (tau=4–5,
coverage 0.91–0.97; its coalition-aware orders are pending the backfill run). phi holds
coverage (0.85–0.97) but at size 5–6, failing the preregistered size cap. **Mistral fails hard
and instructively — tau=inf, coverage capped at 0.78 by the 21% of its errors that escape even
bound-5 families.** The failure is principled (family reach, not ranking), predicted by the
coverage–size law, and already remedied in the machinery: stratified calibration turns
mistral's unreachability into visible abstention instead of silent under-coverage. Pooled
alpha=0.1 stays below target only because mistral is in the pool — the per-model story is the
honest and the strong one.

## The backfill closes the package — scored 2026-07-03

With the coalition-aware orders grafted onto the bound-5 families, **qwen passes the alpha=0.1
guarantee 3/3 seeds on every dataset** (hotpotqa: interaction tau=3–4 coverage 0.91–0.94,
shapley 0.91–0.97 — the predicted upgrade from 2/3 on the contextcite order). phi stays a
size near-miss (tau=4–5); mistral stays the reach boundary. **H4 fails definitively at real
power** (beam vs contextcite on the designed target: qwen -0.03, p=1.0, n=40; phi -0.08,
p=0.909, n=37) — the designed arm ends as a clean, fully-instrumented negative. Bound-5
family structure is stable: irreducibly-shared responsibility 0.34 (0.37 at bound-3),
hierarchy violated 0.55 (0.51).

## Risk-controlled selection — the singleton is undeployable at low error (2026-07-03, exploratory)

Framing single-passage attribution as selection (commit to the top-ranked passage; a false
answer is one that is not a sufficient cause), pooled over 341 wrong cases: the naive
false-answer rate is 0.43, and **no confidence gating on the ContextCite margin pushes the
singleton error below ~0.28-0.34.** A conformal-risk-control threshold therefore abstains
entirely at alpha = 0.1 and 0.2, first commits at alpha ~ 0.3 (~20% answered), and reaches ~85%
answered only at alpha ~ 0.4. So a practitioner who needs a low error on one blamed passage must
abstain or switch to the calibrated set (which meets alpha = 0.1 by coverage) — the
deployment-flavored restatement of the thesis. See `paper/selective.md`; the FDR/eBH variant is
stricter still and selects nothing on this weak signal.

## Mistral row complete + seed replication (2026-07-04)

**The 3x3 H1 grid is 8/9 cells** (only phi/2wiki, phi/musique pending). Mistral, the reach-limited
model, passes H1 on every dataset: hotpotqa 0.79, **2wiki 0.84 [0.76,0.92], musique 0.82
[0.73,0.90]** — lower than qwen (0.92-0.96) but well above 0.50, with H2 failing at its reach
ceiling (coverage 0.77-0.89, tau=inf) on every dataset. Mistral is the consistent, principled
boundary: it has small sufficient sets less often, and where it lacks them the guarantee
correctly abstains.

**Seed replication (qwen/hotpotqa, the headline cell):** H1 seed0 0.96 [0.90,1.0] / seed2 1.00;
ambiguity 0.54 / 0.59. The result is stable across seeds; seed1 (W5) will complete the >=3-seed
set. (results (3)(1).zip was a redundant re-download of the k=10 cell, not scored.)

## What remains to run

Only robustness breadth: the grid fillers and seed replications (W-wave). The preregistered
evidence program is otherwise complete.
