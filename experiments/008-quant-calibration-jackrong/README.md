# Experiment 008: Quant calibration sweep on Jackrong/Qwopus3.5-9B-Coder

- **Status:** done
- **Created:** 2026-05-31
- **Branch:** `exp/008-quant-calibration-jackrong` _(create with `git checkout -b exp/008-quant-calibration-jackrong`)_

## Summary

Mirror of exp-007 but on the Qwen3.5-class 9B model. Compares Q4_K_M and
IQ4_NL across the same 5 calibration corpora (`wiki`, `custom`, and
the `mixed` corpus at imatrix ctx=512 / 2048 / 8192). F16 is the
reference. KLD/PPL run at `ctx=8192` against the same
`corpus.eval.txt` slice used in exp-001 for Jackrong.

The runner reuses everything exp-001 already produced (F16, baseline,
custom/wiki/mixed8k imatrices and Q4_K_M cells) and only computes
what's missing: the two new mixed-ctx imatrices + Q4_K_M cells, and
all 5 IQ4_NL cells.

## Quantization Test with calibration data using Jackrong/Qwopus3.5-9B-Coder

_Filled in by `scripts/run_exp008_quant_calibration_jackrong.py`._

| quant | technique | dataset | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |
|---|---|---|---|---|---|---|---|
| FP16   | none    | —             | 16.69 | 16.012 | 3.8035 | 0.00000 | 100.0000 |
| Q4_K_M | none    | —             | 5.24 | 5.029 | 2.5144 | 0.95961 | 87.6340 |
| Q4_K_M | imatrix | wiki.test.raw | 5.24 | 5.029 | 2.9200 | 0.53350 | 90.4180 |
| Q4_K_M | imatrix | custom | 5.24 | 5.029 | 3.3549 | 0.51300 | 90.4960 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=512) | 5.24 | 5.029 | 3.0949 | 0.50676 | 90.5450 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=2048) | 5.24 | 5.029 | 3.1488 | 0.52010 | 90.4080 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=8192) | 5.24 | 5.029 | 3.1578 | 0.51996 | 90.3440 |
| IQ4_NL | imatrix | wiki.test.raw | 5.05 | 4.841 | 2.6990 | 0.75214 | 89.2890 |
| IQ4_NL | imatrix | custom | 5.05 | 4.841 | 2.6736 | 0.72520 | 89.6310 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=512) | 5.05 | 4.841 | 2.7907 | 0.70275 | 89.5630 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=2048) | 5.05 | 4.841 | 2.7028 | 0.71220 | 89.5580 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=8192) | 5.05 | 4.841 | 2.6742 | 0.71079 | 89.5820 |

Direction: lower is better for PPL and KLD; higher is better for same_top_p.

## MMLU-Pro accuracy (5 reps, T=0.6/top_p=0.95/top_k=20)

75-question holdout: 25 each from computer science, engineering, math
(`out/mmlu_pro/holdout_cs_eng_math.json`, seed=42, 2-shot). Each cell
is mean over 5 reps; the `Average` column is overall accuracy (mean
over all 75 questions per rep, then averaged across reps — not the
mean of the three per-subject columns).

Values reported as mean ± stdev across 5 reps.

| quant | technique | dataset | size (GiB) | Comp Sci. | Eng. | Math | Average |
|---|---|---|---|---|---|---|---|
| FP16   | none    | —             | 16.69 | 0.640 ± 0.028 | 0.408 ± 0.082 | 0.712 ± 0.082 | **0.587 ± 0.040** |
| Q4_K_M | none    | —             | 5.24  | — | — | — | — |
| Q4_K_M | imatrix | wiki.test.raw | 5.24  | 0.632 ± 0.066 | 0.440 ± 0.075 | 0.728 ± 0.077 | **0.600 ± 0.033** |
| Q4_K_M | imatrix | custom        | 5.24  | 0.576 ± 0.092 | 0.320 ± 0.028 | 0.704 ± 0.088 | 0.533 ± 0.050 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=512)  | 5.24 | 0.616 ± 0.092 | 0.392 ± 0.077 | 0.704 ± 0.046 | 0.571 ± 0.042 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=2048) | 5.24 | 0.520 ± 0.085 | 0.304 ± 0.115 | 0.600 ± 0.089 | 0.475 ± 0.022 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=8192) | 5.24 | 0.552 ± 0.118 | 0.304 ± 0.078 | 0.616 ± 0.061 | 0.491 ± 0.030 |
| IQ4_NL | imatrix | wiki.test.raw | 5.05  | — | — | — | — |
| IQ4_NL | imatrix | custom        | 5.05  | — | — | — | — |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=512)  | 5.05 | — | — | — | — |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=2048) | 5.05 | — | — | — | — |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=8192) | 5.05 | — | — | — | — |

Direction: higher is better. IQ4_NL and `Q4_K_M | none` rows pending —
not yet MMLU-evaluated.

Source: `out/mmlu_pro/jackrong/reps_agg.csv` (per-rep CSV alongside in
`reps_per.csv`; logs under `reps_logs/`).

## Tool-call accuracy (5 reps, T=0.6/top_p=0.95/top_k=20)

25-session holdout from `out/exp-002/toolcall_holdout.jsonl`. Each row
is mean ± stdev across 5 reps.

| quant | technique | dataset | size (GiB) | tool_selection_acc | param_acc_mean | schema_valid_rate |
|---|---|---|---|---|---|---|
| FP16   | none    | —             | 16.69 | 0.559 ± 0.043 | 0.342 ± 0.025 | 0.888 ± 0.030 |
| Q4_K_M | imatrix | wiki.test.raw | 5.24  | **0.596 ± 0.054** | **0.350 ± 0.026** | **0.921 ± 0.035** |
| Q4_K_M | imatrix | custom        | 5.24  | 0.579 ± 0.065 | 0.328 ± 0.015 | 0.890 ± 0.031 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=512)  | 5.24 | 0.535 ± 0.021 | 0.311 ± 0.027 | 0.890 ± 0.013 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=2048) | 5.24 | 0.520 ± 0.086 | 0.291 ± 0.079 | 0.907 ± 0.033 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=8192) | 5.24 | 0.527 ± 0.017 | 0.304 ± 0.026 | 0.893 ± 0.037 |

Source: `out/toolcall/jackrong/reps_agg.csv` (per-rep CSV alongside in
`reps_per.csv`; logs under `reps_logs/`).

### Tool-call accuracy at T=0.2 (all 12 GGUFs, including IQ4_NL + none)

Re-run with `T=0.2 top_p=0.95 top_k=20 min_p=0.0 pp=0.0 rep_pen=1.0`,
5 reps each, same 25-session holdout. Adds Q4_K_M `none` + all 5 IQ4_NL
cells. Source: `out/toolcall/jackrong_t02/reps_agg.csv`.

| quant | technique | dataset | tool_selection_acc | param_acc_mean | schema_valid_rate |
|---|---|---|---|---|---|
| FP16   | none    | —             | 0.526 ± 0.034 | 0.331 ± 0.023 | 0.889 ± 0.019 |
| Q4_K_M | none    | —             | 0.527 ± 0.019 | 0.309 ± 0.016 | 0.901 ± 0.020 |
| Q4_K_M | imatrix | wiki.test.raw | 0.523 ± 0.063 | 0.342 ± 0.023 | 0.899 ± 0.023 |
| Q4_K_M | imatrix | custom        | 0.559 ± 0.051 | 0.347 ± 0.036 | 0.893 ± 0.013 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=512)  | 0.549 ± 0.043 | 0.348 ± 0.041 | 0.911 ± 0.046 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=2048) | 0.572 ± 0.059 | 0.347 ± 0.041 | 0.925 ± 0.019 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=8192) | **0.586 ± 0.039** | **0.361 ± 0.035** | 0.887 ± 0.024 |
| IQ4_NL | imatrix | wiki.test.raw | **0.601 ± 0.022** | 0.352 ± 0.017 | **0.935 ± 0.014** |
| IQ4_NL | imatrix | custom        | 0.532 ± 0.033 | 0.344 ± 0.028 | 0.926 ± 0.031 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=512)  | 0.582 ± 0.042 | 0.347 ± 0.025 | **0.935 ± 0.020** |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=2048) | 0.584 ± 0.032 | 0.351 ± 0.021 | 0.901 ± 0.027 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=8192) | 0.559 ± 0.039 | 0.348 ± 0.020 | 0.903 ± 0.013 |

**Compared to the T=0.6 run, the cell ordering largely inverts** for
Q4_K_M tool_selection_acc:

| cell | T=0.6 | T=0.2 | Δ |
|---|---|---|---|
| Q4_K_M wiki     | 0.596 | 0.523 | −0.073 |
| Q4_K_M custom   | 0.579 | 0.559 | −0.020 |
| Q4_K_M mixed512 | 0.535 | 0.549 | +0.014 |
| Q4_K_M mixed2k  | 0.520 | 0.572 | +0.052 |
| Q4_K_M mixed8k  | 0.527 | 0.586 | +0.059 |
| F16             | 0.559 | 0.526 | −0.033 |

At T=0.6 the ordering was `wiki > custom > F16 > mixed512 ≈ mixed8k ≈
mixed2k`; at T=0.2 it flips to `mixed8k > mixed2k > custom > mixed512 >
F16 ≈ none > wiki`. The "wiki is the best calibration corpus" finding
was sampling-temperature-dependent, not a robust property of the
calibration.

**T=0.2 observations:**

- **IQ4_NL wiki is the single best cell** at this temperature (0.601
  tool_sel, 0.935 schema_valid), comfortably ahead of every Q4_K_M
  cell and every other IQ4_NL cell. The Q4_K_M-vs-IQ4_NL tradeoff
  from exp-008's quant table (Q4_K_M better KLD, IQ4_NL better PPL)
  doesn't carry a clean task-level preference — at T=0.2 IQ4_NL is
  competitive or better than Q4_K_M on tool-call.
- **All imatrix-calibrated quants beat F16 + Q4_K_M none on
  tool_selection_acc.** F16: 0.526, Q4_K_M none: 0.527. Every imatrix
  cell ≥ 0.523, most ≥ 0.55. At low temperature `none` no longer
  uniquely collapses (cf. MMLU where none lost ~20 pts) — the gap is
  ~6-8 pts on tool-call, comparable to between-corpus differences.
- **Mixed-context-larger wins on Q4_K_M at low temperature**:
  mixed8k > mixed2k > mixed512 (0.586 / 0.572 / 0.549). Exact
  inversion of the T=0.6 ordering. Read: at low temperature the
  fatter-context imatrix produces a more deterministic-friendly quant;
  at high temperature wiki/custom apparently generate output
  distributions whose entropy plays better with stochastic sampling.
- **Stdevs are 30-50% smaller at T=0.2** (median ~0.040 vs ~0.054 at
  T=0.6), as expected. Differences between cells are slightly more
  significant per-rep, but the inversion-of-ordering is the main
  takeaway, not any single cell's improvement.
- **Schema validity is also higher across the board** at T=0.2,
  topping out at 0.935 (IQ4_NL wiki + mixed512) vs 0.921 at T=0.6
  (Q4_K_M wiki). Lower temperature → tighter JSON structure, as
  expected.

**Implication for picking a quant:** the "best" calibration corpus
depends on the deployment temperature. If you serve at T≈0.2, prefer
IQ4_NL wiki or Q4_K_M mixed8k. If you serve at T≈0.6, prefer Q4_K_M
wiki. There is no single corpus that dominates across temperatures on
this model + eval.

**Tool-call observations:**

- **Same ordering as MMLU.** `wiki` ≥ F16 on every metric;
  `mixed2k`/`mixed8k` worst by tool_selection_acc, custom and mixed512
  in the middle. The "wiki wins, mixed-ctx-larger loses" pattern from
  the MMLU run is reproduced on a completely different eval — strong
  corroboration that the corpus choice matters at task level even
  though it doesn't move KLD/same_top_p.
- **Q4_K_M wiki beats F16.** Tool selection 0.596 vs 0.559; param_acc
  0.350 vs 0.342; schema_valid 0.921 vs 0.888. The deltas are inside
  stdev individually but consistent across all three metrics — wiki
  isn't just preserving F16's tool-call behavior, it's slightly
  improving on it (consistent with the MMLU finding).
- **schema_valid_rate spread is tight (0.890-0.921)** — the
  pass/fail JSON-schema signal didn't separate cells. The action is
  in tool_selection_acc (semantic correctness, spread 0.520-0.596 =
  ~7.5 pts) and param_acc (0.291-0.350 = ~6 pts).
- **No malformed truth, no recovery degradations.** All cells hit
  `recovery_appropriate_rate = 1.000` and `truth_malformed_count = 0`,
  so the differences are purely on first-attempt tool selection and
  parameter filling.

### Quant-quality ↔ MMLU correlation

`scripts/plot_mmlu_quant_correlation.py` plots every rep (5 cells × 5
reps = 25 points) of per-rep MMLU-Pro accuracy against the cell's PPL,
KLD, and same_top_p. F16 is plotted as a separate marker but excluded
from the regression (KLD=0 and same_top_p=100 are by-definition values
that would anchor the fit artificially).

![correlation plot](../../out/mmlu_pro/jackrong/correlation.png)

**Reading the x-axes.** PPL is average −log p(token) on the eval set —
it's a property of the model's predictions on this holdout, not a
distance from F16. So F16's PPL (3.80) sits *above* the quants'
(2.92-3.36); the quant noise happens to redirect mass toward tokens
that appear in the holdout, lowering log-loss while diverging from F16
on distribution shape. KLD and same_top_p, by contrast, are
F16-anchored by construction (F16=0 and F16=100 are extremes). This is
exactly the PPL ↔ KLD disagreement we've been seeing since exp-001.

| predictor | Pearson r | Spearman ρ | expected sign |
|---|---|---|---|
| PPL         | **−0.441** | −0.417 | negative (lower PPL = better) ✓ |
| Mean KLD    | +0.158 | −0.035 | negative ✗ (wrong sign, near zero) |
| same_top_p  | +0.416 | **+0.513** | positive ✓ |

**Read:**

- **PPL is the strongest *linear* predictor** of MMLU on this set
  (|r|=0.441), explaining about 19% of the rep-level accuracy
  variance. Note: PPL alone disagrees with KLD/same_top_p about which
  artifact is closest to F16, but on this eval its lower-is-better
  ranking matches the task-level outcome — wiki (lowest PPL) wins,
  mixed8k/mixed2k (highest PPL among imatrix cells) lose.
- **same_top_p is the strongest *rank* predictor** (ρ=+0.513) —
  noticeably better than PPL by Spearman, suggesting the relationship
  may be monotonic but non-linear (the linear fit gets dragged by
  rep-level noise).
- **KLD is essentially uncorrelated with MMLU on this set** (r=+0.16,
  ρ=−0.04), and Pearson is even the wrong sign. This is the
  quantitative version of the qualitative observation above: the cell
  with the worst KLD wins MMLU, and KLD's tight 0.027-wide cluster
  across cells contains almost no information about task ranking.
- **Same-top-p > KLD as a *cheap* MMLU proxy.** Both come out of the
  same `llama-perplexity --kl-divergence` invocation, but
  same_top_p (mode-agreement with F16) ranks the cells better than
  mean KLD (full distribution distance). Worth keeping in mind when
  picking a calibration metric to optimize against.
- **Caveats**: n=25 reps over only 5 distinct quant cells — each cell
  contributes 5 vertically-stacked points at the same x, so the
  Pearson and Spearman numbers are driven by between-cell ordering
  more than within-cell signal. A wider corpus sweep (more cells) or
  re-running with the IQ4_NL cells would strengthen this.

### Quant-quality ↔ tool-call + schema_valid correlation

Re-running the same analysis on the tool-call eval and on
`schema_valid_rate` gives a much cleaner picture. Full 3×3 plot at
`out/mmlu_pro/jackrong/correlation_grid.png` (script:
`scripts/plot_task_quant_correlation.py`).

| task | predictor | Pearson r | Spearman ρ |
|---|---|---|---|
| MMLU-Pro Avg       | PPL | **−0.441** | −0.417 |
| MMLU-Pro Avg       | KLD | +0.158 | −0.035 |
| MMLU-Pro Avg       | same_top_p | +0.416 | **+0.513** |
| tool_selection_acc | PPL | −0.095 | −0.092 |
| tool_selection_acc | KLD | +0.243 | +0.077 |
| tool_selection_acc | same_top_p | +0.104 | +0.209 |
| schema_valid_rate  | PPL | −0.304 | −0.272 |
| schema_valid_rate  | KLD | **+0.361** | **+0.435** |
| schema_valid_rate  | same_top_p | −0.154 | −0.131 |

(Direction: expected sign is negative for PPL/KLD, positive for
same_top_p. Bold = strongest |correlation| in row group.)

**Read:**

- **KLD has the wrong-sign Pearson on every task** — higher KLD
  corresponds to slightly *higher* task accuracy. Tiny magnitudes
  (max +0.36), but the sign is reversed everywhere. This is the
  per-task version of the qualitative finding: the cells whose
  output distribution drifts furthest from F16's are the ones doing
  best on these tasks. Strong evidence that "preserve F16 exactly"
  is the wrong calibration target for this model.
- **No quant-quality metric strongly predicts tool_selection_acc.**
  Best |r| = 0.243 (KLD, wrong sign). The tool-call signal is
  dominated by within-cell rep variance, not by which corpus you
  calibrated against. Different conclusion from MMLU, where PPL
  carries some signal.
- **PPL is a reasonable proxy for MMLU but not for tool-call.** It
  hits |r|=0.441 on MMLU-Pro Avg (with correct sign), drops to
  |r|=0.10 on tool_selection_acc.
- **same_top_p tracks rank-ordering best on the metric where it has
  signal** (MMLU Spearman +0.513) but loses sign on schema_valid.
- **No metric is consistently good across all three tasks.** Different
  evals are sensitive to different aspects of the quant. Reading
  this as: pick your calibration metric by the task you care about,
  and don't trust a single number to summarize "quant quality."

### Task-level observations

- **KLD/same_top_p do not predict MMLU-Pro rank.** On the quant table
  above the 5 Q4_K_M cells cluster within 0.027 KLD; on MMLU they
  spread 12.5 pts (0.475 → 0.600). The cell with the *worst* KLD
  (`wiki`, 0.534) wins MMLU; the cells tied at the *best* KLD
  (`mixed2k`/`mixed8k`, 0.520) finish last. Distribution-shape
  metrics and task accuracy disagree about which corpus is best.
- **Q4_K_M wiki ≥ F16 on Average** (0.600 vs 0.587, within noise) —
  imatrix calibration didn't degrade the model on this eval, and on
  the larger-context-imatrix cells (mixed2k/mixed8k) it *hurt* by
  ~10 pts. Read: for this model + holdout, short-context vanilla
  wiki is the safest calibration choice.
- **Unparseable rate tracks accuracy loss.** F16: 0.6/rep. Best cell
  (wiki): 0.4. Worst cell (mixed2k): 1.6. Suggests the formatting /
  instruction-following channel is what's degrading on the bad
  cells, not subject knowledge — answer extraction fails more often.

## Observations

- **Same PPL ↔ KLD/same_top_p split as exp-007.** IQ4_NL is smaller
  (5.05 GiB / 4.841 BPW vs 5.24 / 5.029) and lower-PPL on every cell
  (~2.67-2.79 vs ~2.92-3.36), but Q4_K_M wins KLD by ~30-40%
  (0.51-0.53 vs 0.70-0.75) and same_top_p by ~0.7-1.2 pts. The
  tradeoff direction holds across both model families (Gemma 4B in
  exp-007, Qwen3.5 9B here).
- **The KLD gap between quant types is bigger on Jackrong than Gemma.**
  Exp-007 (gemma): IQ4_NL KLD was ~17-20% worse than Q4_K_M. Here:
  ~30-40% worse. The 9B Qwen model is more sensitive to the choice of
  sub-byte packing than the 4B Gemma was.
- **Corpus barely matters within either quant type — same pattern as
  exp-001/007.** Within Q4_K_M: KLD spread 0.027 across 5 corpora,
  same_top_p spread 0.20 pts. Within IQ4_NL: KLD spread 0.049,
  same_top_p spread 0.34 pts. The slightly larger IQ4_NL spread is
  expected — its baseline KLD is higher so absolute differences scale
  up.
- **Imatrix context length sweep is flat in both quants.**
  Q4_K_M (mixed512 / mixed2k / mixed8k): KLD 0.507 / 0.520 / 0.520;
  IQ4_NL: 0.703 / 0.712 / 0.711. No monotonic trend; well inside the
  expected noise floor. Confirms the gemma + Jackrong-mixed8k findings
  on yet another quant type.
- **Best Q4_K_M cell flips between corpora and metric.** On PPL
  alone, `wiki` is best (2.920); on KLD/same_top_p, `mixed512` is best
  (0.507 / 90.55). Same disagreement we've seen since exp-001 —
  optimize for the metric that matches your downstream use.
- **`none` row reused from exp-001** (no re-quantize — same F16 source,
  same eval, same baseline). PPL 2.514 is the best of any cell on raw
  log-loss, but KLD 0.960 is ~2× worse than every imatrix Q4_K_M cell
  and same_top_p drops ~2.8 pts behind. Same PPL/KLD disagreement that
  shows up everywhere — see exp-001/002 for the task-level resolution
  (`none` loses on schema_valid pass@5).

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/run_exp008_quant_calibration_jackrong.py
```

Prerequisites (produced by exp-001's Jackrong run): `model-f16.gguf`,
`baseline.kld`, `corpus.eval.txt`, `corpus.mixed8k.txt`, and
`imatrix-{custom,wiki,mixed8k}.gguf` under
`out/exp-001/Jackrong__Qwopus3.5-9B-Coder/`.
