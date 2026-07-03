# SCoPE results

Regenerate: `python scripts/build_results.py --root <path-to-cells>`

## Natural grid — H1, ambiguity, A3

| cell | rows | bound | parametric | H1 small-set | ambiguity | A3 rate | min-size hist |
|---|--:|--:|--:|---|--:|--:|---|
| 2wiki/natural/qwen | 100 | 3 | 0.21 | 0.95 [0.90, 0.99] | 0.52 | 0.80 | 1:50 2:20 3:5 |
| hotpotqa/natural/mistral | 58 | 3 | 0.09 | 0.72 [0.60, 0.83] | 0.38 | 0.48 | 1:18 2:11 3:9 |
| hotpotqa/natural/phi | 71 | 3 | 0.06 | 0.90 [0.82, 0.97] | 0.54 | 0.66 | 1:37 2:14 3:9 |
| hotpotqa/natural/qwen | 75 | 3 | 0.11 | 0.93 [0.87, 0.99] | 0.51 | 0.69 | 1:52 2:7 3:3 |
| musique/natural/qwen | 100 | 3 | 0.05 | 0.92 [0.85, 0.97] | 0.54 | 0.70 | 1:54 2:23 3:10 |

## Conformal coverage per ranking arm (seeded halves, all seeds shown as min–max)

| cell | arm | alpha | tau | coverage | mean size | n/test |
|---|---|--:|---|---|---|--:|
| 2wiki/natural/qwen | contextcite | 0.1 | 4/5/inf | 0.90–1.00 | 4.0–6.0 | 40 |
| 2wiki/natural/qwen | contextcite | 0.2 | 3 | 0.85–0.93 | 3.0–3.0 | 40 |
| 2wiki/natural/qwen | interaction | 0.1 | 3/6/inf | 0.88–1.00 | 3.0–6.0 | 40 |
| 2wiki/natural/qwen | interaction | 0.2 | 2/3 | 0.78–0.95 | 2.0–3.0 | 40 |
| 2wiki/natural/qwen | shapley | 0.1 | 3/inf | 0.93–1.00 | 3.0–6.0 | 40 |
| 2wiki/natural/qwen | shapley | 0.2 | 2/3 | 0.78–1.00 | 2.0–3.0 | 40 |
| hotpotqa/natural/mistral | contextcite | 0.1 | inf | 0.63–0.67 | 6.0–6.0 | 27 |
| hotpotqa/natural/mistral | contextcite | 0.2 | inf | 0.63–0.67 | 6.0–6.0 | 27 |
| hotpotqa/natural/mistral | interaction | 0.1 | inf | 0.63–0.67 | 6.0–6.0 | 27 |
| hotpotqa/natural/mistral | interaction | 0.2 | inf | 0.63–0.67 | 6.0–6.0 | 27 |
| hotpotqa/natural/mistral | shapley | 0.1 | inf | 0.63–0.67 | 6.0–6.0 | 27 |
| hotpotqa/natural/mistral | shapley | 0.2 | inf | 0.63–0.67 | 6.0–6.0 | 27 |
| hotpotqa/natural/phi | contextcite | 0.1 | 6/inf | 0.82–0.88 | 6.0–6.0 | 34 |
| hotpotqa/natural/phi | contextcite | 0.2 | 4/6 | 0.74–0.88 | 4.0–6.0 | 34 |
| hotpotqa/natural/qwen | contextcite | 0.1 | 4/inf | 0.91–0.94 | 4.0–6.0 | 34 |
| hotpotqa/natural/qwen | contextcite | 0.2 | 2/3 | 0.85–0.94 | 2.0–3.0 | 34 |
| musique/natural/qwen | contextcite | 0.1 | 5/6 | 0.83–0.90 | 5.0–6.0 | 48 |
| musique/natural/qwen | contextcite | 0.2 | 3/4 | 0.75–0.83 | 3.0–4.0 | 48 |
| musique/natural/qwen | interaction | 0.1 | 4/6 | 0.81–0.90 | 4.0–6.0 | 48 |
| musique/natural/qwen | interaction | 0.2 | 3 | 0.79–0.83 | 3.0–3.0 | 48 |
| musique/natural/qwen | shapley | 0.1 | 4/5 | 0.83–0.90 | 4.0–5.0 | 48 |
| musique/natural/qwen | shapley | 0.2 | 2/3 | 0.71–0.81 | 2.0–3.0 | 48 |
| **pooled** | contextcite | 0.1 | inf | 0.89–0.92 | 6.0–6.0 | 181 |
| **pooled** | contextcite | 0.2 | 4 | 0.86–0.89 | 4.0–4.0 | 181 |
| **pooled** | interaction | 0.1 | 5/inf | 0.80–0.89 | 5.0–6.0 | 114 |
| **pooled** | interaction | 0.2 | 3/4 | 0.77–0.86 | 3.0–4.0 | 114 |
| **pooled** | shapley | 0.1 | 5/inf | 0.84–0.89 | 5.0–6.0 | 114 |
| **pooled** | shapley | 0.2 | 3 | 0.81–0.85 | 3.0–3.0 | 114 |

## H5 — leave-one-out vs the intersection of the sufficient family

| cell | agreement | clean n | violations excluded | no set in bound |
|---|---|--:|--:|--:|
| 2wiki/natural/qwen | 0.56 | 16 | 80 | 4 |
| hotpotqa/natural/mistral | 0.40 | 15 | 28 | 15 |
| hotpotqa/natural/phi | 0.59 | 17 | 47 | 7 |
| hotpotqa/natural/qwen | 0.78 | 18 | 52 | 5 |
| musique/natural/qwen | 0.45 | 22 | 70 | 8 |

## Designed arm — construction and extraction verdicts

| cell | H3a designed-sufficient | H3b link vs distractor silent | H4 beam vs contextcite (designed target) |
|---|---|---|---|
| 2wiki/and/qwen | pending (run log) | 0.40 vs 0.27 (fail) | pending |
| hotpotqa/and/mistral | 0.50 [0.33, 0.67] (n=30) | 0.50 vs 0.52 (fail) | 0.20 vs 0.25, p=0.741 (n=20) |
| hotpotqa/and/phi | pending (run log) | 0.41 vs 0.42 (fail) | pending |
| hotpotqa/and/qwen | pending (run log) | 0.26 vs 0.26 (fail) | pending |
| musique/and/qwen | 0.51 [0.37, 0.66] (n=41) | 0.34 vs 0.30 (fail) | 0.05 vs 0.05, p=1.000 (n=20) |
