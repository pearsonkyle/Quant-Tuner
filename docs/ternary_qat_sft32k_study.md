# Continued QAT of a natively-ternary model: the `sft32k` run

**Status: in progress** (see the tail of this file for what is still open). Companion to
`docs/ternary_qat_sft8k_study.md`, which is the *reference run* every number here is read
against. Method and reproduction live in `docs/ternary_qat.md`; the CUDA port, the
measurement tables and the resume runbook live in `docs/qat_32k_handoff.md` §10. This
document is the **observations**.

## What this run is testing

`sft8k-full` changed the model's behaviour without changing its capability: tool-error
rate halved (0.65 → 0.33) but resolve rate stayed 0/10, and the model **lost the ability
to stop** — 7/10 instances hit `max_turns`, 97% of trajectories looping, 2.2M tokens
burned per instance against vanilla's 26k.

Two independent hypotheses, both applied here, neither subsuming the other:

1. **The window was too short to teach termination.** At 8064 only 27% of SWE
   trajectories fit whole, so the model rarely saw a complete task→completion arc. At
   32768 it is **97%** (independently reverified on this build, along with `logs-agents`
   29% → 74%).
2. **The stop token is drowned.** 35,359 terminating `<|im_end|>` targets in 6,071,948 —
   one "stop" decision per 172 "keep going", 0.58% of the loss mass. `--stop-weight 6.0`
   raises it to 3.40%.

The window buys *whole arcs*; the weight buys *salience*. Sessions pack contiguously, so
the target-to-stop ratio is nearly identical at both window sizes (6,060,840/35,046 at
8064 vs 6,071,948/35,359 at 32768) — the longer window does **not** fix the imbalance.

## Setup, and what differs from the reference

| | `sft8k-full` (reference) | `sft32k` (this run) |
|---|---|---|
| window | 8064 | **32768** |
| stop weight | 1.0 | **6.0** |
| steps / epoch | 522 | 613 |
| grad-accum | 4 | **1** |
| tokens / step | 32,256 | 32,768 |
| lr / warmup | 5e-4 cosine, step 30 | 5e-4 cosine, step 30 |
| spike guard | none (did not exist) | **off** (see below) |
| compute | fp32, M4 Max / MPS | **bf16 + fp32 masters**, RTX PRO 6000 |
| cost | 55.4 GPU-h, ~384 s/step | **~2.3 h, 13.9 s/step** |

Tokens-per-step was held constant deliberately so that lr 5e-4 transfers like-for-like
rather than being re-derived. **That is not free**, and it is the one design tension in
this run: holding tokens/step fixed while quadrupling the window forces grad-accum from 4
to 1, so each step is now *one* 32,768-token window (one conversation, highly correlated)
instead of *four* independent 8,064-token ones. Batch size and gradient variance cannot
both be held constant when the window changes. The consequences show up in Observation 2.

**Two defaults were set to match the reference rather than to "improve" on it**, after the
reference's completed report showed its post-warmup loss excursion was a healthy
reorganization (validation improved monotonically through it and ended at its best value),
not a divergence:

- `--warmup-frac 0.05` — 30 of 613 steps, matching step 30 of 522 in both count and
  fraction. An earlier draft used 0.10 to damp the excursion; that was solving a
  non-problem and would have added a third variable.
- `--grad-spike-factor 0` — the reference ran with no guard at all. `GradSpikeGuard`
  documents a warmup-awareness **it does not implement**: `check()` activates on
  `min_history` alone, so at factor 4.0 it goes live at step 21 and would have skipped
  exactly the post-warmup steps — invisibly, since a skipped step leaves no mark on the
  loss curve. Docstring corrected in this branch.

  This runs *against* the reference study's own prescription, which built the guard
  specifically for that excursion and says "clipping did not prevent it, and hid it." The
  two source documents disagree — the completed report calls the excursion healthy, the
  study calls it a 90-step, ~9 GPU-hour cost. This run takes the report's side. If the
  excursion had run away, the intended fallback was a restart with the guard on, then fp32.

## Observation 1 — the excursion was milder and recovered 6× faster

```
             this run              sft8k-full
step 30   loss 1.07  gnorm  6.78   loss 5.49
step 35   loss 6.80  gnorm 17.19   loss 9.80   (both: peak)
step 40   loss 4.39  gnorm 68.18   loss 9.11   VAL 8.67
step 45   loss 4.08  gnorm 14.44
step 50   loss 1.63  gnorm  1.23               (ref reaches 1.34 at step 120)
```

The reference peaked at 9.80 and took **85 steps** to return to 1.34. This run peaked at
6.80 and was at 1.63 by **step 50 — 15 steps**. Validation degraded from 0.8216 to 0.9644
and was back under baseline (0.7851) by step 100; the reference's val hit 8.67 and did not
cross under 1.0 until roughly step 160.

Candidate causes, not separable from one run: the longer window (whole trajectories rather
than fragments), or grad-accum 1. The second predicts the *opposite* sign — one correlated
window should have higher gradient variance than four independent ones — which makes the
window the more likely cause, but this needs the lr 2.5e-4 control to say anything firm.

## Observation 2 — at grad-accum 1 the reported loss is mostly source composition

The printed loss is a mean over the last 5 windows. At accum 1 that is 5 *windows*, not 20,
and consecutive windows are drawn from sources whose losses differ ~5× from each other. The
result is a train-loss series that oscillates 1.4–4.6 with no readable trend, and `gnorm`
spikes (68.18 at step 40, 90.96 at step 55) that turn out to be specific hard windows:

```
step 40  logs=8.27  logs-agents=3.42                gnorm 68.18
step 55  swe=5.72   logs=1.29  logs-agents=1.94     gnorm 90.96
step 65  logs=5.39  logs-agents=1.26                gnorm 23.55
```

Reading a single step's loss here produces false findings — it did, twice, during this run:
a "geometric gnorm runaway" inferred from two points that rolled over at the third, and a
"damage concentrated in `logs`" inferred from one 5-window sample. Averaged over phases,
every source recovered:

| source | pre-excursion (1–30) | excursion (31–65) | after (66–105) |
|---|---|---|---|
| `swe-trajectories` | 1.227 | 3.927 | **0.948** — below baseline |
| `broad-instruct` | 0.771 | 4.179 | 1.040 |
| `logs` | 1.027 | 4.407 | 1.203 |
| `logs-agents` | 1.103 | 1.516 | 1.283 |

`swe-trajectories` — the source the endpoint depends on, and the one the window change was
meant to help — is the only one already below its own pre-excursion level.

**Consequence for the next run:** at accum 1, per-source loss is the readable series and
the aggregate is not. `--grad-accum 2` at this window would halve the noise at the cost of
306 steps and a doubled batch.

## Observation 3 — flip velocity peaks at the same *fraction* of the run

```
checkpoint   mean flip %   Δ (velocity)   recruited   pruned      net
   @ 50        0.0088%      +0.0000          5,947     5,596     +351
   @100        0.1177%      +0.1089        110,890    83,698  +27,192
   @150        0.4396%      +0.3219        414,695   329,075  +85,620
   @200        0.8141%      +0.3745  peak  791,328   637,781 +153,547
   @250        1.1478%      +0.3337
```

The reference peaked at step 200 of 522 (38% through); this run peaks at step 200 of 613
(33%). Different window, different packing, different hardware, same curve shape and
nearly the same location. Everything after the peak is annealing.

Cumulative movement is far ahead of the reference: **1.15% of tracked codes at step 250**
versus **1.19% after all 522** of sft8k-full. (Tensor samples differ — this run samples
layers 0/5/10/15/20/25/30/35, the reference 0/3/6/9/13/16/19/22/26/29/32/35 — so absolute
percentages are not strictly comparable; the shape and per-tensor ordering are.)

## Observation 4 — the model recruits far more than it prunes

791,328 weights switched on against 637,781 switched off by step 200: **net +153,547** dead
weights brought into use, with the ratio stable across every checkpoint. Layer 0's `q_proj`
is the lone net-pruner (density 65.5 → 65.2%), which is also what the reference found.

## Observation 5 — codes move through zero, never across it (reproduced)

`±→∓` transitions total **69** against ~190,000 flips at step 100. The reference's
Observation 3 reproduces exactly on a different corpus and window.

## Observation 6 — the ~30× depth spread reproduces

At step 200: `L0.q_proj` 2.39%, `L5.k_proj` 1.12%, `L35.down_proj` 1.11%, `L15.o_proj`
0.73%, `L20.o_proj` 0.68%, `L25.gate_proj` 0.27%, `L10.v_proj` 0.12%, `L30.up_proj` 0.09%
— a 26× spread, with `q_proj` at layer 0 leading and `up_proj`/`v_proj` barely moving.

Two independent runs now show this. The reference flagged freezing the low-movers as an
untested efficiency win; the case for running that ablation is now stronger.

## Observation 7 — validation cannot see what this run changes

```
step  25   50    75   100   125   150   175   200   225   250
    0.822 0.964 0.897 0.785 0.802 0.818 0.822 0.799 0.788 0.843
```

Flat across 225 steps while flips went 0.009% → 1.15%. This is **not** evidence that
nothing is happening: the val corpus is 81 windows drawn from `logs` and `logs-agents`
only — **no SWE trajectories, no broad-instruct** — so it structurally cannot measure
termination behaviour on agentic tasks, which is the entire object of the run. The
reference's val was similarly flat (0.90–1.06 from step 160 to 320) before creeping to
0.859.

**Fix for the next run:** build the val split to contain the sources under test.

## Cost and the CUDA port

~2.3 h against the reference's 55.4 GPU-h — a 24× reduction, from three separate findings
(all in `qat_32k_handoff.md` §10): the trainer silently selected **CPU** on CUDA; fp32 +
GQA has no fused SDPA kernel so transformers' `enable_gqa=True` dropped every attention
call to the math backend and materialized `[batch, heads, S, S]`; and bf16 is a 5× speedup
here where it was a 28× pessimization on Metal.

bf16's safety for a ternary model was measured, not assumed: it changes **0 of 117M codes**,
because a ternary latent sits at 0 or ±s while `delta = 0.7·mean|W|` sits between them, so
the boundary region is empty by construction. Step-1 `gnorm` parity against fp32 on an
identical window: 1.316 vs 1.307.

## Still open

- [ ] Final flip totals, density deltas and recruit/prune split at step 613.
- [ ] Export to Q2_0 and the **SWE-rebench A/B** — the only measurement that answers whether
      capability moved. **Cannot run on the training box** (unprivileged container: no Docker
      daemon, no `cap_sys_admin`). Bundle and command in `qat_32k_handoff.md` §10.7; holdout
      is `holdout50.jsonl`, 50 instances, **0 overlap** with the 71 trained on.
- [ ] **Primary endpoint: loop fraction**, then `max_turns` exits, tool-error rate, patch
      rate, resolved. At 0/50 the 95% upper bound on resolve rate is ~5.8% (vs ~31% at n=10).
- [ ] The lr **2.5e-4 control** — the reference study's #1 open question, and now a ~2.3 h
      experiment rather than a separate day.
- [ ] Whether the milder excursion is the window or grad-accum 1. Separable with one run.
