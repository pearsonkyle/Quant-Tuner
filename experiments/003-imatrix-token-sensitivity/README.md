# Experiment 003: imatrix token & sequence-length sensitivity

- **Status:** planned
- **Tests hypothesis #1:** investigate the imatrix technique and see if we can improve it using custom data or a new optimization metric
- **Created:** 2026-05-25T20:51:10+00:00
- **Branch:** `exp/003-imatrix-token-sensitivity`

## Summary

Exp-001 and exp-002 swept the **corpus identity** (custom vs wiki) and the
**re-weighting variant** (vanilla vs output-aware) and found both knobs
nearly inert at Q4_K_M / ~5.0 BPW. That leaves two unexplored axes that
also feed into `llama-imatrix`:

1. **Total calibration tokens** — how many tokens of activation statistics
   does the imatrix actually need to converge? Exp-001 used a single point
   (500K).
2. **Per-pass sequence length (`ctx`)** — exp-001 captured at `ctx=512`,
   the llama-imatrix default. Longer windows expose tensors to wider
   inter-token attention patterns; the question is whether that surfaces
   channels the short-window pass misses.

This experiment also revisits the **dataset-mixing** question from exp-001
under one of these new settings: does `custom + wiki` (concatenated) beat
either alone when the calibration budget is held fixed?

Scope: **Jackrong/Qwopus3.5-9B-Coder only** (continuing the exp-002
narrowing), vanilla `E[a²]` re-weighting (no output-aware), Q4_K_M target.
Same held-out eval set as exp-001 / exp-002 (the `holdout` slice of
`logtrain.jsonl` via `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/corpus.eval.txt`).

## Approach

Two sub-experiments, sharing F16 GGUF, F16 KLD baseline, and eval corpus
from exp-001. All cells produce a Q4_K_M GGUF, calibrate via vanilla
`llama-imatrix`, and bench against the shared eval corpus.

### A. Dataset mixing (3 cells, seq=8K, no token-count constraint on wiki)

| Cell | Dataset       | Tokens (target) | `ctx` |
|------|---------------|-----------------|-------|
| A1   | custom        | 500K            | 8K    |
| A2   | wiki.test.raw | ~280K (file)    | 8K    |
| A3   | custom + wiki | ~780K (concat)  | 8K    |

A2 takes whatever `wiki.test.raw` provides (~280K tokens). A3 concatenates
the A1 custom corpus with the A2 wiki file — same content, no resampling.
The comparison is **asymmetric in token count by design**: it answers
"does adding wiki to your custom corpus help?", not "does wiki contribute
more than an equivalent amount of custom?".

### B. Custom-only seq × token grid (9 cells)

| tokens \ ctx | 8K   | 16K  | 20K  |
|--------------|------|------|------|
| 100K         | B11  | B12  | B13  |
| 250K         | B21  | B22  | B23  |
| 500K         | B31  | B32  | B33  |

`target_tokens` controls how much the calibration corpus
(`stratified_pack`) packs from the training slice; `ctx` is passed to
`llama-imatrix` for the capture pass. `per_session_cap=6_000` is held
fixed (same as exp-001) so longer-`ctx` cells don't accidentally get more
per-session content.

**Overlap with A**: A1 == B31 (custom 500K, ctx=8K). It's run once and
the row appears in both tables.

### Reference rows (NOT recomputed — quoted from prior experiments)

| Provenance                  | Cell                                | KLD (mean) | same_top_p |
|-----------------------------|-------------------------------------|------------|------------|
| exp-001 (vanilla, ctx=512)  | Jackrong / custom 500K / ctx=512    | 0.51300    | 90.4960    |
| exp-001 (vanilla, ctx=512)  | Jackrong / wiki / ctx=512           | 0.53350    | 90.4180    |
| exp-001 (no imatrix)        | Jackrong / none                     | 0.95961    | 87.6340    |

Including these makes the `ctx=512 → ctx=8K` jump readable directly off
the B31 row.

## Metrics

Same column shape as exp-001 / exp-002 (size and BPW are invariant at
5.24 GiB / 5.029 since they don't depend on imatrix content).

### A. Dataset mixing

| cell | dataset          | tokens | ctx | PPL | KLD (mean) | same_top_p |
|------|------------------|--------|-----|-----|------------|------------|
| A1   | custom           | 500K   | 8K  |     |            |            |
| A2   | wiki             | ~280K  | 8K  |     |            |            |
| A3   | custom + wiki    | ~780K  | 8K  |     |            |            |

### B. Custom seq × tokens grid

| cell | tokens | ctx | PPL | KLD (mean) | same_top_p |
|------|--------|-----|-----|------------|------------|
| B11  | 100K   | 8K  |     |            |            |
| B12  | 100K   | 16K |     |            |            |
| B13  | 100K   | 20K |     |            |            |
| B21  | 250K   | 8K  |     |            |            |
| B22  | 250K   | 16K |     |            |            |
| B23  | 250K   | 20K |     |            |            |
| B31  | 500K   | 8K  |     |            |            |
| B32  | 500K   | 16K |     |            |            |
| B33  | 500K   | 20K |     |            |            |

Direction: lower PPL/KLD better; higher same_top_p better.

## Observations

_To be filled after running. Things to watch for:_

- _Does same_top_p saturate before 500K tokens, or keep climbing?_
- _Does increasing `ctx` past the llama.cpp default (512) shift KLD at all?_
- _Is `custom + wiki` (A3) any better than the best of A1 or A2 alone, or
  does the longer corpus just dilute the activation statistics?_
- _Memory pressure warnings from `llama-imatrix` at `ctx=20K`?_
- _Any tensors marked as "missing forward stats" or skipped during the
  imatrix capture pass for large `ctx` cells?_

## Analysis

_To be filled. Questions to answer:_

1. _If KLD is flat across the entire B grid → "imatrix is saturated; tokens
   and ctx don't matter at this bit-rate." Combined with exp-001/002, that
   would mean **the imatrix knob is fully tapped out for Q4_K_M on this
   model family** and the remaining error budget is purely about bit-rate
   (move to Q5_K_S)._
2. _If KLD improves monotonically with tokens but flattens at large ctx →
   "tokens matter, ctx doesn't" — practical advice is "use 500K+ tokens
   and the default ctx=512."_
3. _If KLD improves at large ctx → "ctx matters for some channels" — opens
   a new direction (longer-context imatrix becomes a tunable knob)._
4. _If A3 (combined) wins → "data diversity helps even when neither
   individual corpus does" — re-visit the exp-001 finding about corpus
   choice._

## Next steps

This experiment is lower priority than it was at scaffold time —
exp-002's outlier_l4 result substantively closed the schema_valid gap,
which was the main motivator for understanding the corpus/seq/token
axes. Recommended adjustments if and when this runs:

- **Re-run the grid on `outlier_l4` rather than vanilla `E[a²]`.** The
  active hypothesis is now "heavy-tail signal is what matters" — the
  most informative question for sensitivity is whether `√E[a⁴]`
  converges with fewer tokens than `E[a²]` does, or needs the same
  500K. One line change in the driver: pass `variant="outlier_l4"`
  through the calibration step (and ensure forward_stats is built
  per-cell since the corpus / token count changes).
- **Add task-level eval (toolcall reps, 5 reps × 25-session holdout)
  on whatever the bench grid identifies as best.** The original
  KLD-only design predates the finding that KLD and task-level can
  disagree.
- **Skip cells if any axis is clearly saturated.** If 100K tokens
  matches 500K on KLD AND task-level for outlier_l4, the larger-token
  cells are redundant — kill them rather than waste compute.

Original branching paths (preserved for reference):
- If ctx matters → extend to other 2 models at the winning ctx.
- If wiki+custom mixing helps → try other diversity sources (multi-
  language code, formal docs) with token-matched mixing.
