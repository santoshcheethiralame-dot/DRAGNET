# DRAGNET results

Regenerate: `python scripts/build_results.py --root <path-to-cells>`

## Natural grid — H1, ambiguity, A3

| cell | rows | bound | parametric | H1 small-set | ambiguity | A3 rate | min-size hist |
|---|--:|--:|--:|---|--:|--:|---|
| 2wiki/natural/qwen | 100 | 5 | 0.21 | 0.99 [0.96, 1.00] | 0.56 | 0.87 | 1:50 2:20 3:5 4:1 5:2 |
| hotpotqa/natural/mistral | 58 | 5 | 0.09 | 0.79 [0.68, 0.89] | 0.45 | 0.67 | 1:18 2:11 3:9 4:2 5:2 |
| hotpotqa/natural/phi | 71 | 5 | 0.06 | 0.97 [0.93, 1.00] | 0.63 | 0.80 | 1:37 2:14 3:9 4:3 5:2 |
| hotpotqa/natural/qwen | 75 | 5 | 0.11 | 0.96 [0.90, 1.00] | 0.54 | 0.77 | 1:52 2:7 3:3 4:2 |
| musique/natural/qwen | 100 | 5 | 0.05 | 0.97 [0.93, 1.00] | 0.61 | 0.83 | 1:54 2:23 3:10 4:3 5:2 |

## Conformal coverage per ranking arm (seeded halves, all seeds shown as min–max)

| cell | arm | alpha | tau | coverage | mean size | n/test |
|---|---|--:|---|---|---|--:|
| 2wiki/natural/qwen | contextcite | 0.1 | 4/5 | 0.90–1.00 | 4.0–5.0 | 40 |
| 2wiki/natural/qwen | contextcite | 0.2 | 3 | 0.85–0.93 | 3.0–3.0 | 40 |
| 2wiki/natural/qwen | interaction | 0.1 | 3/5 | 0.88–0.97 | 3.0–5.0 | 40 |
| 2wiki/natural/qwen | interaction | 0.2 | 2/3 | 0.78–0.95 | 2.0–3.0 | 40 |
| 2wiki/natural/qwen | shapley | 0.1 | 3/4 | 0.93–1.00 | 3.0–4.0 | 40 |
| 2wiki/natural/qwen | shapley | 0.2 | 2/3 | 0.78–1.00 | 2.0–3.0 | 40 |
| hotpotqa/natural/mistral | contextcite | 0.1 | inf | 0.78–0.78 | 6.0–6.0 | 27 |
| hotpotqa/natural/mistral | contextcite | 0.2 | inf | 0.78–0.78 | 6.0–6.0 | 27 |
| hotpotqa/natural/mistral | interaction | 0.1 | inf | 0.78–0.78 | 6.0–6.0 | 27 |
| hotpotqa/natural/mistral | interaction | 0.2 | inf | 0.78–0.78 | 6.0–6.0 | 27 |
| hotpotqa/natural/mistral | shapley | 0.1 | inf | 0.78–0.78 | 6.0–6.0 | 27 |
| hotpotqa/natural/mistral | shapley | 0.2 | inf | 0.78–0.78 | 6.0–6.0 | 27 |
| hotpotqa/natural/phi | contextcite | 0.1 | 5/6 | 0.85–0.97 | 5.0–6.0 | 34 |
| hotpotqa/natural/phi | contextcite | 0.2 | 4/5 | 0.76–0.91 | 4.0–5.0 | 34 |
| hotpotqa/natural/phi | interaction | 0.1 | 4/5 | 0.79–0.94 | 4.0–5.0 | 34 |
| hotpotqa/natural/phi | interaction | 0.2 | 3/4 | 0.74–0.88 | 3.0–4.0 | 34 |
| hotpotqa/natural/phi | shapley | 0.1 | 4/5 | 0.85–0.97 | 4.0–5.0 | 34 |
| hotpotqa/natural/phi | shapley | 0.2 | 3/4 | 0.79–0.94 | 3.0–4.0 | 34 |
| hotpotqa/natural/qwen | contextcite | 0.1 | 4/5 | 0.91–0.97 | 4.0–5.0 | 34 |
| hotpotqa/natural/qwen | contextcite | 0.2 | 2/3 | 0.85–0.94 | 2.0–3.0 | 34 |
| hotpotqa/natural/qwen | interaction | 0.1 | 3/4 | 0.91–0.94 | 3.0–4.0 | 34 |
| hotpotqa/natural/qwen | interaction | 0.2 | 2/3 | 0.85–0.94 | 2.0–3.0 | 34 |
| hotpotqa/natural/qwen | shapley | 0.1 | 3/4 | 0.91–0.97 | 3.0–4.0 | 34 |
| hotpotqa/natural/qwen | shapley | 0.2 | 2 | 0.85–0.91 | 2.0–2.0 | 34 |
| musique/natural/qwen | contextcite | 0.1 | 4/5 | 0.83–0.96 | 4.0–5.0 | 48 |
| musique/natural/qwen | contextcite | 0.2 | 3/4 | 0.75–0.85 | 3.0–4.0 | 48 |
| musique/natural/qwen | interaction | 0.1 | 4/5 | 0.90–0.92 | 4.0–5.0 | 48 |
| musique/natural/qwen | interaction | 0.2 | 3 | 0.79–0.83 | 3.0–3.0 | 48 |
| musique/natural/qwen | shapley | 0.1 | 4 | 0.92–0.94 | 4.0–4.0 | 48 |
| musique/natural/qwen | shapley | 0.2 | 2/3 | 0.71–0.81 | 2.0–3.0 | 48 |
| **pooled** | contextcite | 0.1 | 5/6 | 0.91–0.95 | 5.0–6.0 | 181 |
| **pooled** | contextcite | 0.2 | 4 | 0.87–0.91 | 4.0–4.0 | 181 |
| **pooled** | interaction | 0.1 | 5 | 0.91–0.93 | 5.0–5.0 | 181 |
| **pooled** | interaction | 0.2 | 3/4 | 0.85–0.92 | 3.0–4.0 | 181 |
| **pooled** | shapley | 0.1 | 4/5 | 0.90–0.95 | 4.0–5.0 | 181 |
| **pooled** | shapley | 0.2 | 3 | 0.85–0.88 | 3.0–3.0 | 181 |

## H5 — leave-one-out vs the intersection of the sufficient family

| cell | agreement | clean n | violations excluded | no set in bound |
|---|---|--:|--:|--:|
| 2wiki/natural/qwen | 1.00 | 12 | 87 | 1 |
| hotpotqa/natural/mistral | 1.00 | 8 | 39 | 11 |
| hotpotqa/natural/phi | 1.00 | 12 | 57 | 2 |
| hotpotqa/natural/qwen | 1.00 | 14 | 58 | 3 |
| musique/natural/qwen | 1.00 | 14 | 83 | 3 |

## Designed arm — construction and extraction verdicts

| cell | H3a designed-sufficient | H3b link vs distractor silent | H4 beam vs contextcite (designed target) |
|---|---|---|---|
| 2wiki/and/mistral | 0.42 [0.30, 0.55] (n=60) | 0.61 vs 0.51 (fail) | 0.45 vs 0.45, p=1.000 (n=11) |
| 2wiki/and/qwen | pending (run log) | 0.40 vs 0.27 (fail) | pending |
| hotpotqa/and/mistral | 0.50 [0.33, 0.67] (n=30) | 0.50 vs 0.52 (fail) | 0.20 vs 0.25, p=0.741 (n=20) |
| hotpotqa/and/phi | pending (run log) | 0.41 vs 0.42 (fail) | 0.16 vs 0.24, p=0.909 (n=37) |
| hotpotqa/and/qwen | pending (run log) | 0.26 vs 0.26 (fail) | 0.03 vs 0.05, p=1.000 (n=40) |
| musique/and/qwen | 0.51 [0.37, 0.66] (n=41) | 0.34 vs 0.30 (fail) | 0.05 vs 0.05, p=1.000 (n=20) |
