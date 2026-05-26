# Experiment 006: outlier_max rescue + heavy-tail signal mixing

- **Status:** planned
- **Tests hypothesis #1:** investigate the imatrix technique and see if we can improve it using custom data or a new optimization metric
- **Branch:** `exp/006-heavy-tail-mixing`

## Summary

Exp-002 (with the mapping fix) gave us two heavy-tail variants whose
behavior diverged sharply when the 48 additional `linear_attn.*`
projections joined the imatrix:

- **`outlier_l4`** (signal: `√E[a⁴]`) — KLD improved (0.504 → 0.499),
  schema_valid mean improved (0.914 → 0.928), every task-level
  metric moved in the right direction. Best variant tested overall.
- **`outlier_max`** (signal: `max|a_c|`) — KLD got *worse* (0.543 →
  0.562), schema_valid pass@5 dropped to 0.895 (worst of any cell).
  Task metrics were essentially flat.

The hypothesis from exp-002's Next steps #4: the `attn_qkv` and
`attn_gate` projections inside the Qwen3-Next linear-attention blocks
have **pathological max|a| spikes** that distort the L1-normalized
per-channel ranking. Outlier_l4's `√E[a⁴]` averages those spikes
across the corpus and is more robust; outlier_max keeps the single
worst observation per channel and over-weights it.

This experiment does two things:

1. **Diagnose** the outlier_max regression directly (Phase A).
2. **Try several rescues** — tensor-class fallbacks (Phase B) and
   mixed-signal variants (Phase C) — to see if a different signal
   shape recovers outlier_max, beats outlier_l4 alone, or surfaces a
   new winner.

Scope: Jackrong/Qwopus3.5-9B-Coder only, custom corpus, Q4_K_M, same
held-out KLD eval and 25-session tool-call holdout as exp-002.

## Approach

All variants are constructed by re-ranking the **existing**
`out/exp-002/Jackrong__Qwopus3.5-9B-Coder/forward_stats.npz` (the
100%-mapping forward pass) plus the vanilla base
`out/exp-001/Jackrong__Qwopus3.5-9B-Coder/imatrix-custom.gguf`. No
new HF forward pass is required — every cell is pure numpy
post-processing. The cells differ only in the per-channel score
function applied per tensor.

### Phase A — Diagnostic (no GGUF; required first step)

`scripts/diagnose_outlier_max.py` loads forward_stats.npz and reports
per-tensor-class summary statistics of `max|a|` and `√E[a⁴]`:

- For each tensor class (`attn_q`, `attn_k`, `attn_v`, `attn_output`,
  `attn_gate`, `attn_qkv`, `ffn_gate`, `ffn_up`, `ffn_down`, `ssm_*`):
  - per-channel `max|a|`: mean, std, max, ratio max/median, ratio
    max/p99 (skew indicator)
  - per-channel `√E[a⁴]`: mean, std, max, same ratios
  - sample size and channel count
- Specifically flags tensor classes where `max/p99 > 10` (extreme
  single-channel outliers) as the suspected culprits.

This output drives whether the Phase B/C cells make sense as defined
or need adjustment. **Run this before launching the rest.**

### Phase B — Tensor-class fallback for outlier_max

| cell | base signal | fallback class               | fallback signal |
|------|-------------|------------------------------|-----------------|
| F1   | `max|a|`    | `attn_qkv` + `attn_gate`     | `E[a²]` (vanilla)|
| F2   | `max|a|`    | `attn_qkv` + `attn_gate`     | `√E[a⁴]` (outlier_l4) |
| F3   | `max|a|`    | all `linear_attn`-mapped     | `E[a²]` (broader fallback) |

If F1 or F2 recovers outlier_max's KLD to ≤ 0.513 (vanilla), the
diagnosis is confirmed — the linear-attn projections are the
culprit. F3 tells us whether the issue is specific to attn_qkv /
attn_gate or to all linear-attn outputs.

### Phase C — Heavy-tail signal mixing

| cell | combiner                                            | rationale |
|------|-----------------------------------------------------|-----------|
| H1   | `max(L1(√E[a⁴]), L1(max|a|))`                       | union of the two heavy-tail signals; tests if max-of-both is more informative than either alone |
| H2   | `0.5·L1(√E[a⁴]) + 0.5·L1(max|a|)`                   | arithmetic blend — averages out single-channel max spikes |
| H3   | `sqrt(L1(√E[a⁴]) · L1(max|a|))`                     | geometric mean (mix_50-style) — channels need to score on **both** signals |
| H4   | `max(L1(E[a²]), L1(√E[a⁴]))`                        | augment vanilla with heavy-tail; tests if E[a²] retains useful signal that outlier_l4 alone loses |

H1–H3 explore whether combining outlier_l4 and outlier_max produces a
better signal than either alone. H4 is the "vanilla-meets-heavy-tail"
combiner — analogous to what `hybrid_custom` does for `E[a²]` and
`‖W‖²·E[a²]`, but with `√E[a⁴]` in place of the output-contribution
term. If H4 wins, exp-004's combiner sweep should be re-run with that
signal pair as default.

### Reference rows (quoted from exp-002, not rerun)

| variant                           | KLD    | schema_valid mean | schema_valid pass@5 | tool_sel mean |
|-----------------------------------|--------|-------------------|---------------------|---------------|
| none                              | 0.960  | **0.939**         | 0.943               | 0.523         |
| vanilla custom                    | 0.513  | 0.905             | 0.932               | 0.549         |
| output_aware                      | 0.526  | 0.883             | 0.927               | 0.539         |
| outlier_l4 (full mapping)         | **0.499** | 0.928          | **0.948**           | 0.539         |
| outlier_max (full mapping)        | 0.562  | 0.901             | 0.895               | 0.540         |
| fp16                              | 0      | 0.911             | 0.906               | 0.534         |

## Metrics

### Phase A — Diagnostic (no metrics; tables generated by the script)

### Phase B + C — Bench (KLD suite)

| cell | description                                | PPL | KLD (mean) | same_top_p |
|------|--------------------------------------------|-----|------------|------------|
| F1   | outlier_max + E[a²] on attn_qkv/attn_gate  |     |            |            |
| F2   | outlier_max + outlier_l4 on attn_qkv/attn_gate |  |            |            |
| F3   | outlier_max + E[a²] on all linear_attn     |     |            |            |
| H1   | max(outlier_l4, outlier_max)               |     |            |            |
| H2   | 0.5·outlier_l4 + 0.5·outlier_max           |     |            |            |
| H3   | sqrt(outlier_l4 · outlier_max)             |     |            |            |
| H4   | max(vanilla, outlier_l4)                   |     |            |            |

### Tool-call rollout (5 reps × 25-session holdout, mean ± stdev + pass@5)

Same shape as exp-002's tables. Filled after running both phases.

## Observations

### Phase A — diagnostic (ran during scaffold validation)

Full output at `out/exp-006/diagnostic.md`. Headline numbers:

**`max|a|` per-channel skew (max/p99) by tensor class:**

| class       | max/p99 | n_tensors |
|-------------|---------|-----------|
| ffn_down    | **12.0×** ⚠ | 32 |
| attn_k/q/v  | 8.98×  | 8 each |
| attn_output | 7.33×  | 8 |
| ssm_out     | 6.52×  | 24 |
| attn_qkv    | 6.42×  | 24 |
| attn_gate   | 6.42×  | 24 |
| ssm_alpha/beta | 6.42× | 24 each |
| ffn_gate/up | 5.90×  | 32 each |

**The initial culprit hypothesis is not supported.** `attn_qkv` and
`attn_gate` (the two non-SSM tensor classes newly added by the mapping
fix) show modest max|a| skew (6.42×) — *lower* than `attn_k/q/v`
(8.98×), `attn_output` (7.33×), or `ffn_down` (12×, the only class
exceeding the 10× threshold). So the regression of outlier_max
when these tensors joined the imatrix is **not** a "pathological
single-channel max spike on linear-attn projections" story.

The actual cause must be more subtle:

- L1-normalization happens per tensor, then signals are compared
  *across* tensors via per-block bit allocation in llama-quantize.
  Adding 48 new tensors with one shape of distribution shifts the
  relative scaling — even modest skew on the new tensors could move
  bit-budget away from previously-prioritized tensors.
- `ffn_down`'s 12× skew was already in the partial mapping, so it
  can't be the new-thing-that-broke-it. But it IS the class most
  likely to suffer from the L1-normalization-meets-per-block-allocation
  interaction — worth instrumenting if a rescue cell fails.

The `√E[a⁴]` signal shows much larger skew across **every** class
(84–470× max/p99). Yet outlier_l4 IMPROVED with the full mapping —
so absolute skew is clearly not what determines whether a heavy-tail
signal helps or hurts. The interaction with L1-norm and per-block
quantization is the load-bearing variable.

### Updated cell recommendations (based on diagnostic)

- **F1, F2, F3 are now less well-motivated** — they assume the new
  tensors have pathological max|a|, which they don't. Run them anyway
  as a clean falsification of the initial hypothesis (~25 min total
  for bench, ~75 min for tool-call eval on 3 cells), but expect null
  or marginal effects.
- **H1–H4 are still well-motivated** — they test signal-shape
  combinations independent of the diagnosis.
- **Consider an additional cell F4**: outlier_max with E[a²] on
  `ffn_down` (the actually-most-skewed class). One-line addition to
  `CELLS` in `scripts/run_exp006.py`. Tests "does isolating the most
  extreme tensor class rescue outlier_max?" — direct alternative
  hypothesis.

### Phase B + C — bench + tool-call (not yet run)

_To be filled when cells execute. Watch for:_

- _Does H4 (vanilla + outlier_l4) beat outlier_l4 alone? If yes,
  E[a²] retains complementary signal even where heavy-tail wins on
  average. That would also matter for exp-004 (combiner sweep)._
- _Do H1–H3 cluster (all similar) or spread (combiner matters)?
  Clustering would suggest the two heavy-tail signals are highly
  correlated; spread would suggest they're capturing different
  information._
- _Surprise outcome: any F cell rescues outlier_max despite the
  diagnostic predicting null — that would mean the mechanism is more
  subtle than per-channel skew. Open a follow-up to figure out why._

## Analysis

_To be filled. Decision tree (updated based on diagnostic):_

1. _If the F cells confirm the diagnostic (null effect on outlier_max
   rescue) → outlier_max's regression is **not** a single-tensor-class
   issue; it's a per-block-allocation interaction effect. Either run a
   second-order experiment (per-block-scale instrumentation) or accept
   outlier_max as unrecoverable on this architecture._
2. _If a surprise F cell wins → the diagnostic missed the mechanism;
   investigate per-block-scale interaction directly before
   generalizing._
3. _If H4 (vanilla + outlier_l4) beats outlier_l4 alone → "always
   augment heavy-tail with vanilla" becomes the recommended pattern;
   exp-004 should use this as a default combiner; recipe promotion
   from exp-002 should switch to H4-style variant rather than plain
   outlier_l4._
4. _If H1–H3 don't improve over outlier_l4 → the heavy-tail signals
   are essentially redundant; outlier_l4 alone is the right default._
5. _If nothing helps → outlier_l4 is at the ceiling for this approach;
   move attention to bit-rate (Q5_K_S) or exp-004 (the broader
   combiner sweep)._

## Next steps

_Conditional on results. Likely:_

- _If a tensor-class-conditional approach wins: extend the rules to
  cover more architecture classes (other hybrid SSM/attention models
  in the OmniCoder/Qwen3.5 family) and verify generalization._
- _If a mixed signal wins: re-test on the other 2 models alongside
  the exp-002 generalization runs._
- _If nothing wins outlier_l4: this experiment is a null result that
  hardens the recommendation "outlier_l4 alone is sufficient at
  Q4_K_M for this workload"._

## Open questions

- **Could a percentile-clipped max (e.g., `p99|a|` instead of `max|a|`)
  rescue outlier_max without a tensor-class fallback?** Computing p99
  per channel requires recording histograms during the forward pass —
  not currently in `forward_stats.npz`. Out of scope for this
  experiment but a natural follow-up if F1/F2 confirm that single-
  channel max spikes are the issue.
- **Does the regression also appear on the other Qwen3-Next-family
  models** (e.g., is this a Jackrong-specific quirk or an
  architecture-family quirk)? Test as part of the exp-002
  generalization runs.
