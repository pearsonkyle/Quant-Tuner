# Experiment 006: outlier_max rescue + heavy-tail signal mixing

- **Status:** done (2026-05-26)
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

All cells at bpw=5.029, size=5.24 GiB. F4 added post-diagnostic.
Reference KLD: outlier_l4 = 0.499, outlier_max = 0.562, vanilla = 0.513.

| cell | description                                | PPL   | KLD (mean) | same_top_p |
|------|--------------------------------------------|-------|------------|------------|
| F1   | outlier_max + E[a²] on attn_qkv/attn_gate  | 3.114 | 0.609      | 89.92      |
| F2   | outlier_max + outlier_l4 on attn_qkv/attn_gate | 2.990 | 0.609  | 89.93      |
| F3   | outlier_max + E[a²] on all linear_attn     | 3.114 | 0.609      | 89.92      |
| F4   | outlier_max + E[a²] on ffn_down            | 3.037 | 0.603      | 89.92      |
| H1   | max(L1(√E[a⁴]), L1(max\|a\|))              | 3.068 | 0.557      | 90.58      |
| H2   | 0.5·L1(√E[a⁴]) + 0.5·L1(max\|a\|)          | 3.171 | 0.531      | 90.58      |
| H3   | sqrt(L1(√E[a⁴]) · L1(max\|a\|))            | 3.220 | 0.535      | 90.49      |
| H4   | max(L1(E[a²]), L1(√E[a⁴]))                 | 3.246 | **0.493**  | **90.73**  |

Note: F1 and F3 are bit-identical because on Qwopus3.5-9B-Coder the
non-SSM `linear_attn`-mapped tensors collapse to exactly `attn_qkv` +
`attn_gate` once `is_ssm` filters out `ssm_*`. F3 added no information
on this model.

### Tool-call rollout (10 reps × 25-session holdout, mean ± stdev)

Ran only the four H-cells; F-cells were clearly dominated on KLD and
not worth ~5 h of additional eval time. Reference rows from exp-002
(same holdout, same sampling, 5 reps there vs. 10 here).

| variant                     | tool_selection_acc | param_acc        | schema_valid     | KLD   |
|-----------------------------|--------------------|------------------|------------------|-------|
| outlier_l4 (exp-002 ref)    | 0.539 ± 0.026      | 0.323 ± 0.020    | 0.928 ± 0.024    | 0.499 |
| outlier_max (exp-002 ref)   | 0.540 ± 0.027      | 0.327 ± 0.017    | 0.901 ± 0.032    | 0.562 |
| H1: max(l4, mx)             | 0.521 ± 0.044      | 0.310 ± 0.027    | 0.926 ± 0.024    | 0.557 |
| H2: 0.5·l4 + 0.5·mx         | **0.530 ± 0.029**  | 0.306 ± 0.017    | 0.911 ± 0.030    | 0.531 |
| H3: sqrt(l4 · mx)           | 0.527 ± 0.026      | 0.318 ± 0.019    | **0.932 ± 0.028**| 0.535 |
| H4: max(E[a²], l4)          | 0.524 ± 0.034      | **0.323 ± 0.017**| 0.920 ± 0.031    | **0.493** |

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

### Phase B — F cells: tensor-class fallback made things worse

All four F-cells regressed KLD from the outlier_max baseline of 0.562
to ≥0.603. F1 (E[a²] fallback on attn_qkv/attn_gate) and F4 (E[a²]
fallback on ffn_down — the actually-most-skewed class) both landed at
~0.60. F2 (outlier_l4 fallback) was no better. **Swapping the signal
on two tensor classes while leaving the rest on outlier_max produced a
worse outcome than uniform application of any single signal.**

This is a stronger result than the predicted null. It implies the
problem is *not* "one tensor class has bad signal shape" — it's that
the per-block bit allocator inside llama-quantize is sensitive to the
*relative* L1-normalized magnitudes across the full tensor set. Mixing
signal types breaks the implicit cross-tensor calibration that uniform
signals preserve. This was foreshadowed in the diagnostic ("L1
normalization happens per tensor, then signals are compared across
tensors via per-block bit allocation") but the experiment turned the
hypothesis into a measurement.

The implication for future signal-engineering work: **don't mix signal
*types* across tensors; only mix signal *values* within a uniform type**
(which is exactly what the H cells do). The hybrid_custom recipe from
the published leaderboard follows this rule by construction — it
combines two values (`E[a²]` and `‖W‖²·E[a²]`) using one consistent
combining function applied to every tensor.

### Phase C — H cells: signal mixing helps KLD but not tool-call

H4 (max(L1(E[a²]), L1(√E[a⁴]))) is the new KLD champion at 0.493,
edging out outlier_l4's 0.499. H2 / H3 cluster around 0.531–0.535,
worse on KLD than outlier_l4 alone but with the best schema_valid_rate
of the experiment (H3 at 0.932). H1 at 0.557 is the weakest H cell.

But on tool-call rollout, the four H cells are statistically
indistinguishable from each other and from the exp-002 outlier_l4
baseline:

- All four H-cell tool_selection_acc means (0.521–0.530) sit inside
  outlier_l4's 1σ band (0.539 ± 0.026). H4's param_acc of 0.323 ties
  outlier_l4 exactly. H3's schema_valid of 0.932 marginally beats
  outlier_l4's 0.928 but well inside both stdevs.
- The ~12% KLD gap between H4 and H1 (0.493 vs 0.557) does not
  translate into a measurable task-level difference at n=10.

## Analysis

The experiment delivered two findings, one structural and one practical.

**Structural finding (high confidence, F cells):** Per-tensor signal
swapping breaks llama-quantize's per-block bit allocation. The
allocator implicitly assumes consistent signal semantics across
tensors, and L1-normalized comparison between different signal types
(max|a| vs. E[a²]) produces a worse outcome than uniform application
of any single signal. This rules out a whole class of "rescue the
under-performing variant by patching one tensor class" strategies and
explains why the published `hybrid_custom` recipe applies one
combining rule uniformly. Worth a brief note in `calibrate/imatrix.py`
near where new variants would be added.

**Practical finding (medium confidence, H cells):** Mixing E[a²] with
heavy-tail signals (H4) produces the best KLD seen on this model and
matches outlier_l4 on every tool-call metric inside the noise floor.
The KLD edge is real but small and doesn't translate to a measurable
task-level win. Recommendation: keep `outlier_l4` as the default for
this workload; H4 is a viable alternative if a downstream consumer
prioritizes KLD specifically, but doesn't justify recipe promotion on
its own. The KLD→tool-call link looks weaker than exp-002's results
suggested — there's an intermediate regime (KLD ~0.49–0.56) where
task metrics flatten out.

Decision-tree outcomes from the original plan:

1. F cells → confirmed *and strengthened* the diagnostic. Per-block
   allocation is the load-bearing mechanism; outlier_max is
   structurally unrecoverable by per-class patching.
2. No surprise F cell wins.
3. H4 ties outlier_l4 on tool-call rather than beating it. Recipe
   promotion not justified.
4. H1–H3 do not improve over outlier_l4 on tool-call.
5. → outlier_l4 is at or near the ceiling for this signal family at
   Q4_K_M on this workload. Attention should move to bit-rate
   (Q5_K_S), exp-004 (broader combiner sweep), or per-block-scale
   instrumentation as a follow-on.

## Next steps

- **Document the F-cell finding in code.** Add a short comment near
  the variant dispatch in `calibrate/imatrix.py` noting "all tensors
  in an imatrix must use the same signal type — mixing signals
  produces worse cross-tensor allocation than uniform application of
  any single signal (see exp-006 F cells)." Prevents future re-runs of
  this experiment.
- **Generalization probe deferred.** Don't run H4 on the other two
  OmniCoder/Qwen3.5 models until exp-004's combiner sweep finishes —
  exp-004 may surface a different winner that supersedes the question.
- **Follow-up exp-007 (speculative):** Instrument per-block scale
  derivation inside `llama-quantize` to measure how the bit allocator
  responds to L1-normalized signal distributions. Would explain the
  F-cell regression mechanistically and inform whether a
  "calibration-aware" allocator could rescue outlier_max. Multi-day
  effort, only worth doing if a downstream user cares about
  outlier_max specifically.

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
