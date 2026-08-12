# Experiment 004: imatrix combiner sweep

- **Status:** planned
- **Tests hypothesis #1:** investigate the imatrix technique and see if we can improve it using custom data or a new optimization metric
- **Branch:** `exp/004-imatrix-combiner-sweep`

## Summary

Exp-002 found that `output_aware` (defined as
`max(L1(E[a²]), L1(‖W[:,c]‖²·E[a²]))` per tensor) was a wash on KLD and
*hurt* schema_valid_rate by 5.6 pts vs no calibration. One plausible
culprit isn't the additional output-contribution term itself — it's the
**combiner**:

- The elementwise `max` is aggressive: a channel important on either
  signal gets the high score, so the combined distribution flattens
  toward uniform and erodes the natural sparsity of the original
  `E[a²]`.
- L1-normalizing both signals before combining equalizes their dynamic
  range — but `‖W‖²·E[a²]` naturally has a different scale, and that
  asymmetry might encode useful information about which signal should
  dominate per tensor.

This experiment **holds the two input signals fixed** (`E[a²]` and
`‖W‖²·E[a²]`) and sweeps the combiner / normalization. If a different
combiner shape recovers the schema_valid that output-aware lost, the
fix is the combine rule, not the underlying idea.

Scope: Jackrong/Qwopus3.5-9B-Coder only, custom corpus, Q4_K_M, same
held-out eval and 25-session tool-call holdout as exp-002.

## Approach

Reuse exp-001's `imatrix-custom.gguf` (vanilla base) and F16. The
combiner variants are built inline in `scripts/run_exp004.py` (no
changes to `src/quant_tuner/calibrate/imatrix.py`) so the experiment
stays self-contained. If a variant wins, promote it into the main
module afterward.

### Cells (5 new, with vanilla + output_aware + none as reference)

| cell | combiner                                                              | normalization | notes |
|------|-----------------------------------------------------------------------|---------------|-------|
| C1   | `sqrt(E[a²] · ‖W‖²·E[a²])` per tensor                                 | L1 both       | == `mix_50` (already in `calibrate/imatrix.py:264`) — geometric mean instead of `max` |
| C2   | `0.25·E[a²] + 0.75·‖W‖²·E[a²]`                                        | L1 both       | mostly output term |
| C3   | `0.50·E[a²] + 0.50·‖W‖²·E[a²]`                                        | L1 both       | equal arithmetic blend |
| C4   | `0.75·E[a²] + 0.25·‖W‖²·E[a²]`                                        | L1 both       | mostly input term |
| C5   | `max(E[a²], ‖W‖²·E[a²])`                                              | **none**      | same `max` as output_aware but without L1-norm — preserves natural scale (output term dominates) |

### Reference rows (quoted from exp-001 / exp-002, not rerun)

| label                            | KLD   | same_top_p | tool_sel | schema_valid |
|----------------------------------|-------|------------|----------|--------------|
| imatrix / custom / vanilla       | 0.513 | 90.50      | 0.549    | 0.905        |
| imatrix / custom / output_aware  | 0.526 | 90.37      | 0.539    | 0.883        |
| imatrix / custom / outlier_l4    | _exp-002 ext_ | _exp-002 ext_ | _exp-002 ext_ | _exp-002 ext_ |
| none                             | 0.960 | 87.63      | 0.523    | 0.939        |

## Metrics

### Bench (KLD suite)

| cell | combiner        | size (GiB) | BPW   | PPL | KLD (mean) | same_top_p |
|------|-----------------|------------|-------|-----|------------|------------|
| C1   | mix_50          |            |       |     |            |            |
| C2   | α=0.25 blend    |            |       |     |            |            |
| C3   | α=0.50 blend    |            |       |     |            |            |
| C4   | α=0.75 blend    |            |       |     |            |            |
| C5   | max (no norm)   |            |       |     |            |            |

### Tool-call rollout (5 reps × 25-session holdout, same as exp-002)

| cell | combiner       | tool_sel_acc | param_acc | schema_valid | continuation_match |
|------|----------------|--------------|-----------|--------------|--------------------|
| C1   | mix_50         |              |           |              |                    |
| C2   | α=0.25 blend   |              |           |              |                    |
| C3   | α=0.50 blend   |              |           |              |                    |
| C4   | α=0.75 blend   |              |           |              |                    |
| C5   | max (no norm)  |              |           |              |                    |

## Observations

_To be filled. Things to watch for:_

- _Does any combiner cell recover schema_valid_rate to ≥ 0.939 (the `none` level)?_
- _Does the α-blend show a monotone trend (more output term → worse schema)?_
- _Does C5 (max without L1-norm) behave very differently from output_aware
  (max with L1-norm)? If yes, the normalization is the problem, not the
  combiner shape._
- _Any tensors with imatrix scores that go to zero or NaN under the new
  combiners? (mix_50's geometric mean can collapse if one factor is zero.)_

## Analysis

_To be filled. Key decisions:_

1. _Best schema_valid wins?_ → That combiner becomes a candidate for
   promotion into `calibrate/imatrix.py` as a new named variant.
2. _Monotone α trend?_ → Output-contribution signal is the source of the
   regression; recommend dropping it.
3. _Nothing helps?_ → Combiner is not the issue; move on to exp-005
   (block-aware aggregation) or to bit-rate (Q5_K_S).

## Next steps

The framing has shifted since this experiment was scaffolded.
`outlier_l4` (heavy-tail signal `√E[a⁴]`) already substantively closed
the schema_valid gap on Jackrong (pass@5 0.948 vs `none` 0.943). So
this experiment's most informative version isn't the original 5 cells
swept over `E[a²]` and `‖W‖²·E[a²]` — it's an extended set that uses
**the heavy-tail term as the second signal**.

Recommended adjustments before running:

- **Extend to a 6th and 7th cell** that combine `E[a²]` with
  `√E[a⁴]` instead of `‖W‖²·E[a²]`. The interesting question now is
  whether a combined `E[a²]` + heavy-tail signal beats `outlier_l4`
  alone. One `mix_50`-style geometric mean and one `α=0.5` arithmetic
  blend would cover most of the value.
- **Always include `outlier_l4` and `none` as reference rows** in the
  results tables (not just vanilla and output_aware) so the comparison
  is to the current best, not just the historical baseline.

If a cell beats outlier_l4 on schema_valid pass@5 *and* matches it on
KLD, that becomes the new candidate winner. Otherwise the conclusion
is "outlier_l4 alone is the right signal — the combiner doesn't help."
Either way, exp-002's Next steps #1 (generalize across the other two
models) still has to happen before any of this gets promoted to the
default recipe.
