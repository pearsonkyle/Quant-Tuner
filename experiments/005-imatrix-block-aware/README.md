# Experiment 005: block-aware imatrix aggregation

- **Status:** planned
- **Tests hypothesis #1:** investigate the imatrix technique and see if we can improve it using custom data or a new optimization metric
- **Branch:** `exp/005-imatrix-block-aware`

## Summary

Q4_K_M quantizes weights in **super-blocks of 256 input channels** (8
sub-blocks of 32, each with its own 6-bit scale, all multiplied by a
shared FP16 super-block scale). The current imatrix has **one value
per input channel** — but llama-quantize ultimately collapses 256 of
those values into one scale decision per super-block. If a single
high-importance channel sits inside a super-block of otherwise-quiet
channels, the entire super-block gets extra precision allocated by
that single channel's signal.

Two failure modes follow:

1. **Single-channel spikes drive 256-channel decisions.** A noisy
   E[a²] spike inflates a whole super-block's scale, wasting bits on 255
   weights that don't need them.
2. **The per-channel imatrix doesn't reflect what llama-quantize actually
   sees.** If we aggregate the imatrix to match the quantization block
   structure *before* writing the GGUF, we control exactly what signal
   the quantizer optimizes against.

This experiment tests whether **explicit block-aware aggregation** (max
or mean over 256-channel groups, broadcast back to per-channel) recovers
schema-validity that exp-002's `output_aware` lost, or improves on
vanilla without losing anything.

Scope: Jackrong/Qwopus3.5-9B-Coder only, custom corpus, Q4_K_M, same
held-out eval and 25-session tool-call holdout as exp-002.

## Approach

Aggregation is a post-processing step on an existing imatrix GGUF.
Pseudocode per tensor:

```python
# scores has shape (n_in,), n_in is always a multiple of 256 for Q4_K_M
groups = scores.reshape(n_in // 256, 256)
agg    = groups.max(axis=1)            # or .mean(axis=1)
out    = np.repeat(agg, 256)           # broadcast back to per-channel
```

Block size is intentionally **256** (Q4_K super-block) by default; cell
E5 tests block=32 (Q4 sub-block) for comparison.

SSM tensors pass through unchanged (no `y = W·a` structure; the
quantization for these tensors isn't Q4_K-blocked the same way).

Implementation: inline in `scripts/run_exp005.py`. If a variant wins,
promote into `calibrate/imatrix.py` as a named post-processor afterward.

### Cells

| cell | base imatrix              | aggregation | block | rationale |
|------|---------------------------|-------------|-------|-----------|
| E1   | vanilla custom (exp-001)  | max         | 256   | preserve per-block "any-channel-important" signal |
| E2   | vanilla custom (exp-001)  | mean        | 256   | smooth out single-channel spikes; quiet blocks stay quiet |
| E3   | output_aware (exp-002)    | max         | 256   | does block-max rescue output_aware's regression? |
| E4   | output_aware (exp-002)    | mean        | 256   | does block-mean rescue it? |
| E5   | vanilla custom (exp-001)  | max         | 32    | finer granularity sanity check (sub-block alignment) |

### Reference rows (quoted from exp-001 / exp-002, not rerun)

| label                            | KLD   | same_top_p | tool_sel | schema_valid |
|----------------------------------|-------|------------|----------|--------------|
| imatrix / custom / vanilla       | 0.513 | 90.50      | 0.549    | 0.905        |
| imatrix / custom / output_aware  | 0.526 | 90.37      | 0.539    | 0.883        |
| none                             | 0.960 | 87.63      | 0.523    | 0.939        |

## Metrics

### Bench (KLD suite)

| cell | base | aggregation | block | PPL | KLD (mean) | same_top_p |
|------|------|-------------|-------|-----|------------|------------|
| E1   | vanilla | max  | 256 |     |            |            |
| E2   | vanilla | mean | 256 |     |            |            |
| E3   | output_aware | max  | 256 |     |            |            |
| E4   | output_aware | mean | 256 |     |            |            |
| E5   | vanilla | max  | 32  |     |            |            |

### Tool-call rollout (5 reps × 25-session holdout)

| cell | base | aggregation | block | tool_sel_acc | param_acc | schema_valid | continuation_match |
|------|------|-------------|-------|--------------|-----------|--------------|--------------------|
| E1   | vanilla | max  | 256 |              |           |              |                    |
| E2   | vanilla | mean | 256 |              |           |              |                    |
| E3   | output_aware | max  | 256 |     |           |              |                    |
| E4   | output_aware | mean | 256 |     |           |              |                    |
| E5   | vanilla | max  | 32  |              |           |              |                    |

## Observations

_To be filled. Things to watch for:_

- _Does block-max equal block-mean on vanilla? If yes, single-channel
  spikes are rare in the existing imatrix and aggregation is mostly
  pass-through._
- _Does block-mean on output_aware recover schema_valid? If yes, the
  problem was "single channel dominating a block", not the output term
  itself._
- _Does aggregation at 32 (E5) differ meaningfully from 256 (E1)? If
  E5 == E1, the 256-granularity is the right alignment._
- _Any tensors where n_in is not divisible by 256? (Shouldn't happen
  for Q4_K_M-eligible layers, but verify.)_

## Analysis

_To be filled. Decision tree:_

1. _E2 (block-mean on vanilla) beats vanilla on KLD or schema_valid?_ →
   Single-channel spikes were a real problem. Promote block-mean as a
   pre-write step.
2. _E3 or E4 recover schema_valid above vanilla?_ → output_aware's
   regression was about per-channel granularity, not the output signal
   itself. Block-aware + output_aware is the new candidate winner.
3. _All E cells behave identically to their bases?_ → Aggregation has
   no effect; llama-quantize is already doing this internally, and the
   imatrix is being interpreted at block granularity already. Move to
   bit-rate.

## Next steps

The original motivation for this experiment — closing the schema_valid
regression that imatrix calibration introduced — is largely resolved
by `outlier_l4` (see exp-002 Next steps). Recommended adjustments
before running:

- **Add cells E6 and E7 that aggregate the `outlier_l4` imatrix** (max
  and mean over 256-channel super-blocks). If block-aware aggregation
  has any independent effect, it should compound with the heavy-tail
  signal; if it doesn't, that strengthens the conclusion that
  llama-quantize already does this aggregation internally.
- **Drop or deprioritize E3 and E4** (block-aware applied to
  `output_aware`) — output_aware is the worst variant on every metric,
  so improving it isn't strategically valuable. Replace with
  outlier_l4-based cells.
- **Read `vendor/llama.cpp/ggml/src/ggml-quants.c` Q4_K scale logic
  before running.** If llama-quantize already does per-block
  aggregation when computing scales, every cell here will match its
  base and the experiment is moot — better to know that upfront than
  to interpret a uniform null result.

Lower-priority outcomes (only worth pursuing if the experiment runs
AND a block-aware variant beats outlier_l4):

- Upstream the winning aggregation as a post-write filter in
  `calibrate/imatrix.py`, applied to all variants.
- Re-test promotion of the recipe (exp-002 Next steps #2) using the
  block-aware variant instead of plain outlier_l4.

## Open question

It's possible llama-quantize already aggregates per-channel imatrix
values to block granularity internally when computing scales. If so,
explicit aggregation here is a no-op and all E cells will match their
bases. Worth one round of reading
`vendor/llama.cpp/ggml/src/ggml-quants.c` (search for `iq` / `Q4_K`
scale computation) before drawing strong conclusions from a null result.
