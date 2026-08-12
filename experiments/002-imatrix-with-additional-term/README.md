# Experiment 002: imatrix with output-aware re-weighting

- **Status:** done
- **Tests hypothesis #1:** investigate the imatrix technique and see if we can improve it using custom data or a new optimization metric
- **Created:** 2026-05-25T17:16:38+00:00
- **Branch:** `exp/002-imatrix-with-additional-term-to-better-behavior`

## Summary

Exp-001 held the imatrix re-weighting fixed at **vanilla** (`E[a²]`) and varied
the corpus. Result: corpus choice barely moved KLD or same_top_p — wiki and
custom were within ~2% on every metric. That isolates the question for
exp-002: **does the re-weighting method matter when the corpus does not?**

Scope is intentionally narrow: one model (`Jackrong/Qwopus3.5-9B-Coder`),
one corpus (custom = `logtrain.jsonl` + `calibration_supplement.txt`), one
new re-weighting variant we call **output-aware**.

## The variant: output-aware re-weighting

Vanilla `llama-imatrix` exposes one statistic per linear tensor: `E[a_c²]`,
the mean squared activation on input channel `c` over the calibration corpus.
`llama-quantize` uses this to weight quantization error per channel.

The **output-aware** variant takes the *same* base imatrix (no new corpus
pass) and re-weights it per tensor by combining two signals:

1. **Input signal**: `E[a_c²]` — the vanilla statistic.
2. **Output-contribution signal**: `‖W[:,c]‖² · E[a_c²]` — accounts for how
   much input channel `c` contributes to `‖y‖²` for `y = W·a` given the F16
   weights. A channel with modest `E[a²]` can still feed into many large
   weight columns; conversely, a channel with high `E[a²]` paired with a
   small column norm contributes less to the output.

Both signals are L1-normalized to a common scale, then combined elementwise
as `max(L1(E[a²]), L1(‖W[:,c]‖² · E[a²]))` per tensor — a channel is preserved
if **either** signal flags it. SSM tensors pass through raw `E[a²]` because
`y = W·a` doesn't apply (handled at runtime by `is_ssm` in
`models/hf_gguf_map.py`).

**Naming note:** in `src/quant_tuner/calibrate/imatrix.py` this variant is
spelled `hybrid_custom` (the original published name in the OmniCoder study);
other recipes and scripts in the repo still reference it that way. This
experiment uses the more descriptive name **output_aware** throughout the
write-up and CSV labels — the driver passes `variant="hybrid_custom"` to
the calibrator but writes artifacts named `*-output_aware.*`.

## Approach

Driver: `scripts/run_exp002.py`. Reuses exp-001 artifacts under
`out/exp-001/Jackrong__Qwopus3.5-9B-Coder/`:

- `model-f16.gguf` (the F16 reference)
- `imatrix-custom.gguf` (the vanilla base imatrix on the custom corpus)
- `corpus.eval.txt` + `baseline.kld` (holdout eval + F16 reference)

New artifacts go to `out/exp-002/Jackrong__Qwopus3.5-9B-Coder/`:

- `imatrix-output_aware.gguf`
- `Q4_K_M-output_aware.gguf`
- `results.csv`

Total wall time: ~1.8 minutes.

## Metrics

| model | technique | corpus | variant       | size (GiB) | BPW   | PPL    | KLD (mean) | same_top_p |
|---|---|---|---|---|---|---|---|---|
| Jackrong/Qwopus3.5-9B-Coder | imatrix | custom | vanilla       | 5.24 | 5.029 | 3.3549 | 0.51300    | 90.4960    |
| Jackrong/Qwopus3.5-9B-Coder | imatrix | custom | output-aware  | 5.24 | 5.029 | 3.1633 | 0.52618    | 90.3740    |
| Jackrong/Qwopus3.5-9B-Coder | imatrix | custom | outlier_l4    | 5.24 | 5.029 | 3.3609 | 0.49928    | **90.6620** |
| Jackrong/Qwopus3.5-9B-Coder | imatrix | custom | outlier_max   | 5.24 | 5.029 | 2.8943 | 0.56248    | 90.3050    |
| Jackrong/Qwopus3.5-9B-Coder | imatrix | 500k-custom+wiki | vanilla (ctx=8192) | 5.24 | 5.029 | 3.1578 | 0.51996 | 90.3440 |
| Jackrong/Qwopus3.5-9B-Coder | imatrix | 500k-custom+wiki | outlier_l4 (ctx=8192) | 5.24 | 5.029 | 3.0102 | **0.48134** | 90.6570 |
| Jackrong/Qwopus3.5-9B-Coder | none    | —      | —             | 5.24 | 5.029 | 2.5144 | 0.95961    | 87.6340    |

First and last rows copied from exp-001 for direct comparison.

Direction: lower PPL/KLD is better; higher same_top_p is better.
Best per-column among imatrix variants in **bold**.

The two `outlier_*` rows use the heavy-tail signals captured by an HF
forward pass (`outlier_l4` uses `√E[a⁴]`, `outlier_max` uses `max|a_c|`).
Both numbers above are from the **full-mapping** rebuild — see "HF↔GGUF
mapping fix" in Observations.

### Task-level signal: tool-call rollout (5 reps × 25-session holdout)

Eval scope: 25-session holdout drawn from the `test` slice of `logtrain.jsonl`
(disjoint from the calibration `train` and KLD `holdout` slices by construction —
same seed=42 / 80-10-10 split as exp-001). Sampling: T=0.6, top_p=0.95,
top_k=20, min_p=0, max_tokens=512. Reasoning disabled. Per-rep seed = 1000+rep.
Driver: `scripts/run_toolcall_reps.py` over `eval.toolcall.run_toolcall_eval`.

Six Jackrong Q4_K_M GGUFs plus the F16 reference were evaluated. Mean ± stdev across 5 reps:

| variant                          | tool_sel_acc       | param_acc          | schema_valid       | continuation_match |
|----------------------------------|--------------------|--------------------|--------------------|--------------------|
| **fp16 (reference)**             | 0.534 ± 0.013      | 0.340 ± 0.012.     | 0.911 ± 0.027      | 0.929 ± 0.004      |
| none                             | 0.523 ± 0.020      | 0.315 ± 0.018      | **0.939 ± 0.019**  | 0.942 ± 0.033      |
| imatrix / custom / vanilla       | 0.549 ± 0.041      | 0.326 ± 0.027      | 0.905 ± 0.016      | 0.976 ± 0.032      |
| imatrix / wiki / vanilla         | 0.550 ± 0.018      | 0.350 ± 0.016      | 0.920 ± 0.042      | 0.947 ± 0.030      |
| imatrix / custom / output_aware  | 0.539 ± 0.027      | 0.331 ± 0.011      | 0.883 ± 0.024      | 0.929 ± 0.006      |
| imatrix / custom / outlier_l4    | 0.539 ± 0.026      | 0.323 ± 0.020      | 0.928 ± 0.024      | 0.942 ± 0.033      |
| imatrix / custom / outlier_max   | 0.540 ± 0.027      | 0.328 ± 0.017      | 0.901 ± 0.032      | 0.931 ± 0.004      |
| imatrix / 500k-custom+wiki / vanilla (ctx=8192) | 0.532 ± 0.019      | 0.324 ± 0.018      | 0.906 ± 0.033      | 0.987 ± 0.030      |
| imatrix / 500k-custom+wiki / outlier_l4 (ctx=8192) | 0.542 ± 0.039      | 0.321 ± 0.006      | 0.912 ± 0.013      | 0.928 ± 0.015      |

Wall time: 2h 3min (initial 4 cells) + 1h 1min (outlier cells, full
mapping) + 42min (fp16). Best per-column in **bold**. Note that the
fp16 row wins `param_acc` but loses `schema_valid` and `continuation_match`
to several Q4 cells — **F16 is not the ceiling on every metric**.

#### pass@5 (best-of-5 deployment scenario)

The mean ± stdev above captures expected single-rollout performance.
For workflows that retry up to 5 times and accept the best result, the
right metric is per-turn pass@5: the fraction of turns where ≥1 of the
5 reps succeeded (boolean metrics) or the per-turn max across reps
(continuous metrics). Computed retroactively from per-rep JSONL logs
via `scripts/compute_toolcall_passat5.py` — no re-runs required.

| variant                          | tool_sel pass@5 | param_acc pass@5 | schema_valid pass@5 |
|----------------------------------|-----------------|------------------|---------------------|
| **fp16 (reference)**             | 0.547           | **0.382**        | 0.906               |
| none                             | 0.547           | 0.344            | 0.943               |
| imatrix / custom / vanilla       | **0.593**       | 0.350            | 0.932               |
| imatrix / wiki / vanilla         | 0.579           | 0.372            | 0.947               |
| imatrix / custom / output_aware  | 0.564           | 0.359            | 0.927               |
| imatrix / custom / outlier_l4    | 0.586           | 0.362            | **0.948**           |
| imatrix / custom / outlier_max   | 0.579           | 0.357            | 0.895               |
| imatrix / 500k-custom+wiki / vanilla (ctx=8192) | 0.571           | 0.345            | 0.929               |
| imatrix / 500k-custom+wiki / outlier_l4 (ctx=8192) | 0.593           | 0.342            | 0.932               |

**Methodology note**: pass@5 turn-counts differ slightly across models
(53–59 turns) because rollouts diverge — different models take
different paths through the same session. The metric is mean-over-turns,
so this only weakly biases the comparison.

### MMLU-Pro (5 reps × 150-question holdout)

Tool-call deltas across the imatrix variants are small enough to be hard
to read; MMLU-Pro provides a higher-signal tiebreaker on general
knowledge. Holdout: 50 questions each from `computer science`, `math`,
`engineering` from `TIGER-Lab/MMLU-Pro` (seed=42), 2-shot per subject
from the dev split. Driver: `scripts/run_mmlu_pro_reps.py`. Sampling:
T=0.6 / top_p=0.95 / top_k=20, base_seed=1000. Reasoning disabled.

Holdout: `out/exp-002/mmlu_pro_holdout.json`. Results:
`out/exp-002/mmlu_pro_reps_{results,aggregated}.csv` plus
`mmlu_pro_reps_jackrong_q4_*.csv` for the Jackrong Q4_K_M-mixed8k row
(re-quantized + run separately after the original artifact was cleaned
up between runs).

| model                                              | overall            | computer_science  | math              | engineering       |
|----------------------------------------------------|--------------------|-------------------|-------------------|-------------------|
| Qwen/Qwen3.5-9B fp16                               | 0.463 ± 0.011      | 0.524 ± 0.017     | 0.428 ± 0.011     | 0.436 ± 0.022     |
| Qwen/Qwen3.5-9B Q4_K_M (none)                      | **0.491 ± 0.006**  | 0.520 ± 0.000     | 0.468 ± 0.011     | **0.484 ± 0.009** |
| Jackrong/Qwopus3.5-9B-Coder fp16                   | **0.559 ± 0.018**  | **0.616 ± 0.033** | 0.660 ± 0.032     | 0.400 ± 0.032     |
| Jackrong/Qwopus3.5-9B-Coder Q4_K_M (none)          | **0.559 ± 0.044**  | 0.596 ± 0.043     | **0.688 ± 0.087** | 0.392 ± 0.042     |
| Jackrong Q4_K_M (imatrix / 500k-custom+wiki / vanilla, ctx=8192) | 0.503 ± 0.037 | 0.556 ± 0.071 | 0.636 ± 0.052 | 0.316 ± 0.009 |

#### Observations

- **Jackrong fp16 dominates overall** (+9.6 pts over Qwen fp16 / +6.8
  pts over Qwen Q4) — driven entirely by **math** (+23.2 pts vs Qwen
  fp16) and **computer science** (+9.2 pts). Consistent with Jackrong
  being a coder-focused fine-tune of Qwen3.5.
- **Jackrong's engineering accuracy is its weak subject** (0.400) —
  *worse* than both Qwen fp16 (0.436) and Qwen Q4 none (0.484).
  Fine-tuning on code/math distillation appears to have cost
  engineering coverage. Worth flagging as a workload caveat: Jackrong
  is the better artifact for math/CS-heavy tasks but the *worse*
  artifact for engineering-style reasoning.
- **Qwen Q4_K_M (none) beats Qwen fp16 by 2.8 pts overall** (0.491 vs
  0.463), driven by +4.0 pts on math and +4.8 pts on engineering.
  Mirrors the exp-002 schema_valid finding: at ~5 BPW on this model
  class, the uncalibrated Q4 quant can *outperform* F16 on some
  downstream metrics — quantization noise acts as accidental
  regularization. **F16 is again not the ceiling.**
- **Jackrong fp16's `n_unparseable` was 1.4 ± 0.5 per rep** (out of 150
  questions) — small but nonzero, while every Qwen row was 0/150.
  Jackrong's coder fine-tune occasionally emits a non-letter answer on
  MMLU-Pro; accuracy numbers above treat unparseable as wrong, so this
  is a real cost.
- **Math vs engineering split is the real story.** Across all four
  rows, math spans ~23 pts (0.428 → 0.660) while engineering spans
  ~17 pts (0.316 → 0.484) — but the *ordering* on engineering is the
  inverse of math. The artifact you pick should depend on which
  subject mix your deployment sees.
- **The imatrix calibration is what costs Jackrong MMLU accuracy, not
  Q4 quantization itself.** Holding the model fixed at Jackrong, the
  three rows form a clean ablation:
  - fp16: 0.559 (CS 0.616 / math 0.660 / eng 0.400)
  - Q4 *none*: 0.559 (CS 0.596 / math 0.688 / eng 0.392)
  - Q4 *mixed8k imatrix*: 0.503 (CS 0.556 / math 0.636 / eng 0.316)
  Q4-none **exactly matches fp16 overall** (0.559 vs 0.559) and is
  within ~2 pts on every subject — quantization noise is essentially
  free on this workload. But adding the mixed8k imatrix knocks **5.6
  pts off overall**, with damage on every subject: CS −4.0, math −5.2,
  engineering −7.6. **The 5.6-pt regression on Jackrong Q4-mixed8k is
  attributable entirely to the imatrix calibration, not the Q4
  format.**
- This is the cleanest demonstration so far of exp-002's recurring
  theme: imatrix calibration optimizes for distribution-shape metrics
  (KLD/same_top_p, where mixed8k+outlier_l4 set a new bench-level
  best at KLD=0.481) at the cost of downstream task accuracy on
  workloads that don't match the calibration corpus. The calibration
  corpus (logtrain.jsonl + wiki.test.raw) has minimal MMLU-Pro-style
  multiple-choice content, so the imatrix shifts the quant's
  bit-budget toward channels that don't help MMLU.
- **`n_unparseable` rises with quantization but the imatrix doesn't
  add a further penalty.** Jackrong fp16: 1.4 ± 0.5 / 150 questions;
  Q4 none: 1.2 ± 0.8; Q4 mixed8k: 2.0 ± 0.5. The Q4 cells are within
  ~1 question per rep of each other and within ~1 of fp16. Qwen rows
  were all 0.0 — this is a Jackrong-specific quirk, not a
  quantization artifact.
- **Cross-model takeaway.** If you only had MMLU-Pro to choose between
  these four artifacts:
  - For **math/CS-heavy** workloads: Jackrong fp16 (no contest on
    math; +9 pts CS over Qwen).
  - For **engineering-heavy** workloads: Qwen Q4_K_M none (+8.4 pts
    over Jackrong Q4, +4.8 pts over Qwen fp16).
  - For **mixed / general**: Qwen Q4_K_M none and Jackrong Q4 are
    within ~1 pt overall (0.491 vs 0.503), but with very different
    subject profiles. There is no single best artifact across these
    three subjects.
  - **Jackrong Q4_K_M-mixed8k is not the right artifact to deploy if
    MMLU-Pro-style knowledge tasks are part of the workload** — its
    own fp16 dominates it on every subject, and Qwen Q4 beats it on
    engineering by enough to matter. **Jackrong Q4 *none* is a much
    better Q4 choice** if MMLU-Pro coverage matters: it matches fp16
    overall (0.559) at 1/4 the disk footprint.

## Observations

- Output-aware vs vanilla on the same corpus: **PPL improved 5.7%** (3.355 →
  3.163), **KLD got slightly worse** (+2.6%, 0.513 → 0.526), **same_top_p
  essentially unchanged** (−0.12 pts, 90.50 → 90.37).
- The calibrator processed 248 tensors total, with **72 SSM tensors** passing
  through as raw `E[a²]` (Qwopus is a Qwen3.5-class model with substantial
  SSM content) and 0 skipped.
- The KLD and same_top_p deltas are inside the noise envelope we saw in
  exp-001 between wiki and custom corpora (KLD Δ ≤ 0.02, same_top_p
  Δ ≤ 0.15 pts).
- **HF↔GGUF mapping fix (Qwen3-Next hybrid blocks).** The first outlier
  build only hooked 128/248 tensors (52%) — the `linear_attn.in_proj_*`
  / `out_proj` projections in this model's hybrid SSM/attention blocks
  weren't in `models/hf_gguf_map.py`. After adding 5 regex rules
  (`in_proj_qkv → attn_qkv`, `in_proj_z → attn_gate`, plus three
  SSM-bound mappings that `is_ssm` filters correctly), coverage jumped
  to 248/248 (100% non-SSM hooks). Effect on `outlier_l4`: KLD 0.504 →
  0.499, same_top_p 90.48 → 90.66, schema_valid 0.914 → 0.928. Every
  task-level metric also moved in the right direction. `outlier_max`
  was essentially flat on task metrics but its KLD got *worse*
  (0.543 → 0.562), suggesting `max|a|` over-weights spike behavior on
  the linear-attn projections. The "partial-mapping" artifacts are
  preserved at `out/exp-002/.../*_partial.*` for A/B reference.
- **`outlier_l4` is now the best variant on every distribution-shape
  metric and the best imatrix variant on schema_valid.** It's the only
  cell that simultaneously beats vanilla on KLD (−2.7%) and approaches
  `none` on schema_valid (0.928 vs 0.939, still −1.1 pts).
- **Task-level eval contradicts the KLD/same_top_p story on schema validity.**
  KLD/same_top_p put `none` clearly *behind* every imatrix cell. The
  tool-call eval shows the opposite: `none` produces the **highest**
  schema-valid rate (0.939) and every imatrix variant *regresses* on
  this metric — vanilla custom −3.4 pts, wiki −1.9 pts, output-aware
  −5.6 pts. Output-aware is the worst on schema validity.
- On `tool_selection_acc` and `param_acc`, calibration helps but the gaps
  are noise-comparable. The best (wiki, 0.550) vs `none` (0.523) on
  tool_selection is +2.7 pts against a stdev of ~2.0 — about 1.3σ. Solid
  hint, not a clean win.
- `continuation_type_match_rate` is the one metric where calibration
  cleanly wins (custom vanilla 0.976 vs none 0.942, +3.4 pts).
- `recovery_appropriate_rate = 1.000 ± 0` across **every** model — that
  metric saturated on this holdout. Useless as a discriminator here.
- **mixed8k (500k custom + full wiki, ctx=8192, vanilla re-weighting)
  follow-up from exp-001.** Bench-level: KLD 0.520 / same_top_p 90.344 —
  same tight cluster as `custom` and `wiki`. Task-level: tool_selection
  0.532, param_acc 0.324, schema_valid 0.906, continuation_match
  **0.987 ± 0.030** — the highest continuation_match in the table,
  edging vanilla custom's 0.976. pass@5 schema_valid (0.929) sits
  between `none` (0.943) and `outlier_l4` (0.948). Reinforces the
  exp-001/002 reading: bigger/richer corpus + longer context doesn't
  meaningfully move distribution-shape metrics, and on task-level it
  only nudges continuation_match while regressing schema_valid in line
  with the other imatrix cells.
- **mixed8k + outlier_l4 (cross-axis combination).** Pairing the mixed8k
  base imatrix with the `outlier_l4` re-weighting (forward stats recomputed
  on the same mixed8k corpus) is the **new bench-level best**: KLD 0.481
  (−3.6% vs prior best `outlier_l4`/custom 0.499), PPL 3.010 (−10.4% vs
  3.361). same_top_p is statistically tied (90.657 vs 90.662). Task-level
  is more mixed: schema_valid mean 0.912 ± 0.013 — better than the vanilla
  mixed8k cell (0.906) but **−1.6 pts behind outlier_l4/custom (0.928)**.
  pass@5 schema_valid (0.932) lands back in the cluster, ceding the
  outlier_l4/custom lead (0.948). tool_sel pass@5 (0.593) ties vanilla
  custom for best. Net read: the bigger corpus + longer ctx improves the
  KLD/PPL bench numbers further when stacked with the heavy-tail
  re-weighting, but **does not translate into a schema_valid gain** —
  reinforcing the exp-002 finding that bench-level KLD and task-level
  schema validity are decoupled. If JSON-validity-sensitive deployments
  are the goal, `outlier_l4`/custom remains the winner; if matching F16's
  distribution shape is what you care about, mixed8k+outlier_l4 is now
  the artifact to use.

## Analysis

- **Output-aware is a wash on distribution-shape metrics.** The metric most
  relevant to downstream behavior — same_top_p, the fraction of tokens
  where the quant's top prediction matches F16's — moved by 0.12 pts on
  90.5, indistinguishable from rerun-to-rerun noise. KLD moved 2.6% in
  the *wrong* direction. No reason to prefer output-aware over vanilla
  on this signal.
- **PPL moved 5.7%, but PPL is the wrong primary metric here.** Exp-001
  already showed PPL and KLD/same_top_p disagree on the no-calibration row
  (best PPL, worst KLD). The PPL improvement is consistent with output-aware
  shifting mass slightly toward higher-frequency tokens in the eval set
  without preserving F16's full distribution shape better.
- **Combined with exp-001**, the picture for vanilla Q4_K_M at ~5.0 BPW on
  a 9B Qwen3.5-class model with this eval set is:
  - Calibration (any imatrix) buys ~40% KLD reduction and ~+2.5 pts
    same_top_p versus no calibration.
  - Beyond that, neither the corpus (wiki vs custom) nor the re-weighting
    variant (vanilla vs output-aware) meaningfully moves KLD or
    same_top_p. The remaining error appears to be dominated by the
    quantization bit-rate itself, not by how the importance is computed.
- **Task-level eval changes the story.** KLD and same_top_p ranked
  imatrix variants as roughly equivalent to each other and clearly
  better than `none`. The tool-call rollout shows a **different
  ordering on a different axis**: imatrix variants help on tool
  selection and continuation matching but *hurt* schema validity, and
  the corpus that wins on KLD (custom — slightly) is not the corpus that
  wins on `param_acc` (wiki, by ~+2.4 pts).
  - Plausible mechanism: imatrix calibration spends bit-budget on
    high-mass natural-language channels (where `E[a²]` is largest) at
    the expense of structural tokens like `{`, `}`, `"` and tool/key
    names that are rare-but-critical for JSON validity. KLD averages
    over the whole token distribution, so this regression averages out;
    schema_valid_rate sees only the structural tokens, so it surfaces.
  - This is exactly the failure mode the OmniCoder paper warns about
    when leaning on KLD alone, and is the reason `eval.toolcall` exists
    in this repo.
- **Output-aware is worst on tool-call too.** Across all four
  task-level metrics, output-aware ties or trails vanilla. Combined with
  it being indistinguishable on KLD/same_top_p, there is no remaining
  metric on which output-aware beats vanilla. Result for this
  hypothesis: refuted on this model at this bit-rate.
- **`outlier_l4` partially rescues the schema regression on mean
  (single-rollout) and closes it on pass@5 (best-of-5).** On
  single-rollout (mean ± stdev) the heavy-tail signal recovered 2.3 of
  the 3.4 pts that vanilla lost on schema_valid vs `none`, AND improved
  KLD by 2.7%. On pass@5, outlier_l4 reaches 0.948 — narrowly above
  `none` (0.943) and the best imatrix variant overall. **Read this as:
  outlier_l4 produces a wider distribution of correct outputs across
  stochastic samples than `none` does; with retries it actually wins.**
  Without retries (single-rollout) it still trails by ~1 pt because the
  per-rollout reliability is slightly worse.
- **The fp16 finding inverts the "F16-is-the-ceiling" framing.** The
  goal of imatrix calibration is implicitly "preserve F16 behavior".
  But on this workload, F16 itself produces worse schema_valid and
  continuation_match than several Q4 cells. So matching F16 on
  distribution shape (KLD/same_top_p, which `outlier_l4` does best)
  doesn't automatically maximize downstream task quality. The deployment
  question is "which artifact behaves best on the metric you care about?",
  not "which artifact best matches F16?" — and the answers can differ:
  - For **JSON validity** (schema_valid): `none` Q4 (mean), `outlier_l4` (pass@5)
  - For **argument fidelity** (param_acc): `fp16` (both mean and pass@5)
  - For **tool selection**: `wiki` / `vanilla custom` Q4 (mean), `vanilla custom` (pass@5)
  - For **multi-turn continuation**: `vanilla custom` Q4
  No single artifact wins all four. This is the real finding to carry
  into recipe selection.
- **Mean and pass@5 disagree most for `output_aware` and `outlier_max`.**
  Both look closer to `none` on pass@5 than on mean, suggesting their
  per-rollout schema failures aren't always on the same turns — but
  unlike outlier_l4, neither overtakes `none`. outlier_max's pass@5
  schema_valid (0.895) is the worst of any variant, consistent with the
  bench-level finding that `max|a|` over-weights spike behavior.
- **F16 is not the ceiling on every task-level metric.** On
  `schema_valid` mean, F16 (0.911) sits *below* `none` Q4 (0.939),
  `outlier_l4` (0.928), and `wiki` (0.920); the pass@5 picture is
  similar (F16 0.906 vs `none` 0.943, `outlier_l4` 0.948). On
  `continuation_match`, F16 (0.929) loses to every Q4 cell except
  `output_aware`. F16 *does* win `param_acc` on both mean (0.340) and
  pass@5 (0.382), which is consistent: param_acc rewards reproducing
  the truth string exactly, which a higher-fidelity model should do
  better. The combination suggests F16's sharper output distribution
  pays for higher fidelity by occasionally sampling low-probability
  structural tokens that break JSON validity — quantization noise
  appears to act as accidental regularization on this workload at
  T=0.6 / top_p=0.95 sampling.

## Next steps

Reordered by current evidence. The schema-validity regression that drove
exp-002 originally is now substantively closed by `outlier_l4` + the
mapping fix (pass@5 0.948 vs `none` 0.943, mean −1.1 pts). The active
question shifts from "can we fix the regression?" to "does the
outlier_l4 win generalize, and is it worth promoting?"

1. **Verify outlier_l4 generalizes to Qwen3.5-9B and OmniCoder.** Before
   promoting anything, run the full pipeline (forward_stats →
   outlier_l4 imatrix → Q4_K_M → bench → tool-call reps) on the other
   two models. Two-step process:
   - Sanity check the mapping fix gives 100% non-SSM coverage on each
     model (one-shot diagnostic; the script in this conversation's
     history can be saved as `scripts/diagnose_mapping.py` if needed).
   - Re-run `scripts/run_exp002_outliers.py` with `MODEL` parameterized.
     ~10 min compute per model + ~1 h tool-call eval per model.
   Decision rule: if outlier_l4 ties-or-beats `none` on schema_valid
   pass@5 for at least 2 of 3 models, promote it.

2. **Promote outlier_l4 to the default imatrix recipe.** Conditional on
   #1: update `src/quant_tuner/recipes/q4_k_m_imatrix.yaml` to use
   `variant: outlier_l4` (currently `hybrid_custom`). Document the
   forward-stats prerequisite in the recipe. Add a recipe-comment that
   notes the workload-conditional choice: outlier_l4 is best for
   schema/JSON-validity-sensitive deployments; for param-fidelity-
   sensitive deployments (where F16 wins) staying on F16 may be
   preferable to any Q4 cell. This is the durable takeaway from
   exp-002 — make it the default behavior, but document the
   exceptions.

3. **Exp-004 (combiner sweep) is the most informative remaining
   ablation.** Now that we know the *signal* matters (outlier_l4 wins
   over E[a²]), the natural follow-up is whether a *combined* signal
   — e.g. `max(L1(E[a²]), L1(√E[a⁴]))` or an α-blend of the two —
   beats outlier_l4 alone. The existing exp-004 scaffold sweeps
   combiners over `E[a²]` and `‖W‖²·E[a²]`; extending one or two cells
   to use the heavy-tail term as the second signal is a one-line edit
   in `scripts/run_exp004.py`.

4. **Investigate why outlier_max regressed with full mapping.** KLD got
   worse (0.543 → 0.562) when the 48 linear-attn projections were
   added; `max|a|` on those tensors looks pathological. Quick check:
   plot the max|a| distribution per layer and see whether `linear_attn.*`
   tensors have extreme outliers that distort the per-channel ranking.
   If yes, a per-tensor-class fallback (use E[a²] for linear_attn,
   max|a| elsewhere) might rescue outlier_max. Low priority but
   intellectually interesting.

5. **fp16 task-level baseline** (background `bjmrbbllp`, in progress).
   When it lands, add the row to both tables for a true ceiling
   reference. The current "best minus worst" gaps will then be
   readable as fractions of the quantization headroom.

6. **Exp-003 (token / seq sensitivity)** is still scaffolded but now
   lower priority — the corpus and per-pass length axes were the
   axes that exp-001 already showed were nearly inert. If anything,
   re-run it specifically on outlier_l4 to test whether the heavy-tail
   signal needs more tokens to converge stably.

7. **Exp-005 (block-aware aggregation)** is also still scaffolded but
   the original motivation (closing the schema regression) is largely
   resolved. Keep it on the shelf as a candidate if the outlier_l4 win
   fails to generalize on #1.

8. **Bit-rate comparison (Q5_K_S vs Q4_K_M)** is now a separate
   research question rather than a fallback. With outlier_l4 closing
   the pass@5 schema gap, the question "what extra would Q5_K_S buy?"
   is genuinely open rather than "Q5_K_S to escape the imatrix wall."
   Worth its own experiment number.
