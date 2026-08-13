# Continued QAT of a natively-ternary model: the `sft8k-full` run

**Status: in flight** (step 305/522 at the time of writing). This is the running record of
what the run's telemetry shows, and the source document for a write-up. Numbers here are
extracted from the training log by `scripts/parse_qat_log.py`; raw series live in
`out/exp-058/telemetry/{steps,val,flips}.csv`.

Method, hyperparameters and the reproduction command are in `docs/ternary_qat.md` — this
document is the *observations*, not the how-to.

## Why this run is interesting

A natively-ternary model stores `w = s·c`, `c ∈ {−1,0,+1}`. Post-hoc quantization
calibration (imatrix/AWQ/GPTQ) is a structural no-op on it: the "F16" checkpoint is a
lossless container for weights that are already ternary, so there is no quantization error
to recover. The only lever is continued training with the ternarization in the loop.

That makes the interesting quantity not the loss but **which codes change**, because a
ternary model can lower its loss two ways:

- **scale drift** — the per-group scale `s` moves, every code stays put. Cheap, and mostly
  cosmetic: it rescales what the model already computes.
- **code flips** — `c` actually changes. This is the only way the model's *function* changes
  in a way that survives export to a 2-bit GGUF.

A run can show a falling loss with ~0% flips (measured previously at lr 3e-4 on a smaller
corpus). That looks like learning and isn't. So every observation below is reported against
flip telemetry, not loss.

## Setup

| | |
|---|---|
| base | `prism-ml/Ternary-Bonsai-8B` (native ternary, 36 layers) |
| corpus | universal SFT export, all 5 sources, uncapped — 2088 windows × 8064 tok = 19.4M tok |
| sources | CLI logs (931 win), agent logs (989), SWE trajectories (106), broad-instruct (52), red-team refusals (10) |
| objective | masked CE on assistant/tool-call spans + the terminating `<\|im_end\|>` |
| schedule | 1.0 epoch = 522 steps, lr 5e-4 cosine, 5% warmup, grad-accum 4 |
| hardware | M4 Max 128 GB, all-36 layers, fp32 latents, Adafactor, ~375 s/step |

## Observation 1 — the run diverged at peak LR, then fully recovered

```
step  25   loss 1.06   lr 4.62e-4     last healthy step of warmup
step  30   loss 5.49   lr 5.00e-4     peak LR reached
step  35   loss 9.80   lr 5.00e-4     peak loss
step  40   loss 9.11              VAL 8.67
step  80   loss 5.91              VAL 6.06
step 120   loss 1.34              VAL 1.10     recovered
step 160                          VAL 0.96     best
step 305   loss 1.01              VAL 1.06
```

The lr 5e-4 "sweet spot" came from a prior run on 12 Python SWE trajectories at window
4096. This corpus is ~150× larger with 5 heterogeneous sources at window 8064. **The LR did
not transfer**, and the failure mode was not a gentle plateau — it was a 9× loss excursion
that took 90 steps (~9 h) to unwind.

Two things make this worth reporting rather than hiding:

1. **It recovered completely** under nothing but cosine decay — no intervention, no restart.
   Val ended *below* where it started the excursion. A ternary model's discrete codes appear
   to give it somewhere to fall back to; the excursion did not destroy the model.
2. **The flips it produced during the excursion were not wasted.** Flip velocity kept rising
   through and after the recovery (Observation 4), so the codes disturbed at high loss were
   re-settled rather than frozen in a damaged state.

The honest caveat: we cannot prove the run would not have been *better* without the
excursion, because there is no lower-LR control at this corpus size. That control is the
obvious next experiment.

**We could not diagnose the divergence in flight, because gradient norm was never
recorded.** The trainer clips to 1.0 but never reported the pre-clip value — the single
number that would have said whether this was a bad batch or a systemically too-high LR.
That gap is now fixed (see "Instrumentation added").

## Observation 2 — the shipped model is ~34–42% zeros, and it is exactly ±symmetric

Baseline census of the shipped weights (`scripts/ternary_distribution.py census`):

| tensor | −1 | 0 | +1 |
|---|---|---|---|
| `35.mlp.down_proj` | 33.0% | **34.1%** | 33.0% |
| `0.self_attn.q_proj` | 32.7% | 34.5% | 32.7% |
| `3.self_attn.v_proj` | 32.6% | 34.9% | 32.6% |
| `9.mlp.up_proj` | 31.5% | 37.1% | 31.5% |
| `19.mlp.gate_proj` | 29.5% | 40.9% | 29.6% |
| `26.self_attn.k_proj` | 29.3% | 41.4% | 29.3% |
| `29.self_attn.v_proj` | 29.0% | **42.0%** | 29.0% |

Two facts fall out immediately. The −1 and +1 populations match to within 0.1pp in every
tensor — the code distribution is symmetric to a degree that is clearly structural, not
incidental. And **the zero fraction is not uniform**: the first and last layers are the
densest (34%) while the mid-stack sits near 42%. About 38% of this model's weights
contribute nothing, and how much varies systematically with depth.

That zero band is the capacity training has to work with, which makes its movement — not
the loss — the thing to watch.

## Observation 3 — codes move *through* zero; they essentially never cross it

Decomposing every code change over 300 steps into recruitment (`0→±`), pruning (`±→0`) and
direct sign crossing (`+1→−1` or `−1→+1`):

| tensor | zero-frac start → now | Δpp | recruited | pruned | **sign-crossed** |
|---|---|---|---|---|---|
| `26.self_attn.k_proj` | 41.38 → 40.68% | −0.703 | 44,650 | 15,151 | **1** |
| `22.mlp.down_proj` | 39.37 → 38.89% | −0.486 | 357,672 | 113,092 | **0** |
| `19.mlp.gate_proj` | 40.90 → 40.58% | −0.316 | 212,225 | 53,319 | **6** |
| `32.mlp.gate_proj` | 39.36 → 39.09% | −0.265 | 192,386 | 59,006 | **15** |
| `9.mlp.up_proj` | 37.05 → 36.92% | −0.131 | 115,024 | 48,921 | **0** |
| `6.mlp.gate_proj` | 35.24 → 35.17% | −0.062 | 85,543 | 54,538 | **0** |
| `13.self_attn.q_proj` | 36.91 → 36.92% | +0.002 | 245,673 | 246,024 | **932** |
| `35.mlp.down_proj` | 34.05 → 34.16% | +0.102 | 478,498 | 530,039 | **9** |
| `0.self_attn.q_proj` | 34.52 → 35.04% | +0.526 | 266,330 | 354,620 | **965** |

**Direct sign crossings are ~0.** Against hundreds of thousands of recruit/prune events,
the largest sign-crossing count is 965 (0.15% of that tensor's changes) and six tensors
record literally zero. This is a cumulative comparison against step 0 — a weight that
started at +1 and now sits at −1 *would* be counted — so this is not an artifact of the
sampling interval. In 300 steps of training, a ternary weight that wants to change polarity
does not go `+1 → −1`; it parks at zero and stays there.

That reframes what "1.86% of codes flipped" means. It is not the model rewiring signs; it
is the model **opening and closing gates**, with zero as the resting state in between.

The second split is net direction, and it tracks tensor role:

- **MLP and k/v projections densify** — recruitment outruns pruning 2–7×, so the zero band
  shrinks. The model is switching on capacity that shipped dead.
- **`q_proj` churns at constant-or-rising sparsity** — layer 0 recruits 266k and prunes
  355k, a net *increase* in zeros, and layer 13 is within 0.002pp of break-even on 246k
  events in each direction. Query projections are the one place the model is doing
  wholesale substitution rather than accumulation.

A single flip-percentage would have shown `q_proj` at 28× `v_proj` and implied v_proj was
idle, when in fact the two are doing categorically different things.

Figures live in one document, `out/exp-058/telemetry/report.html` — seven panels plus
the step-0/step-N distribution tables. Regenerate with:

```bash
python scripts/parse_qat_log.py TRAIN.log --out out/exp-058/telemetry
python scripts/ternary_distribution.py census --model out/exp-057/model \
    --tensors out/exp-058/telemetry/flips.csv --out out/exp-058/telemetry/census.csv
python scripts/ternary_distribution.py census --latents .../trained_latents.pt \
    --tensors out/exp-058/telemetry/flips.csv --out out/exp-058/telemetry/census_latest.csv
PYTHONPATH=scripts python scripts/qat_report.py --telemetry out/exp-058/telemetry \
    --census out/exp-058/telemetry/census.csv \
    --latest out/exp-058/telemetry/census_latest.csv --latest-step 325 \
    --window 8064 --grad-accum 4 --out out/exp-058/telemetry/report.html
```

`ternary_distribution.py` is data only (census + trajectory); all plotting is in
`qat_report.py`.

## Observation 4 — flip velocity peaks and decays, in depth order

Cumulative flip % cannot distinguish a tensor that settled early from one still
oscillating. The per-checkpoint *delta* can. Velocity (flip-% added per 25 steps):

```
0.self_attn.q_proj    .00 .01 .24 .51 .56 .54 .43 .50 .31 .28 .20 .14   peak @125
35.mlp.down_proj      .00 .01 .07 .07 .12 .33 .34 .28 .28 .20 .18 .13   peak @175
22.mlp.down_proj      .00 .00 .00 .01 .03 .09 .13 .14 .16 .14 .12 .10   peak @225
3.self_attn.v_proj    .00 .00 .00 .00 .00 .02 .03 .04 .04 .04 .04 .03   peak @250
```

Every tracked tensor has passed its velocity peak and is decaying — the run is
**converging, not thrashing**. And the peak is ordered:

| peak step | tensors |
|---|---|
| 125 | `0.q_proj` |
| 175 | `35.down_proj` |
| 200 | `13.q_proj`, `29.v_proj` |
| 225 | six tensors (mid-stack MLP + `26.k_proj`) |
| 250 | `3.v_proj` |

The first attention layer moves first and settles first; the mid-stack MLPs peak ~100 steps
later; the low-velocity v_projs are the last to move at all. Learning appears to propagate
from the input-side attention outward, with each tensor's activity switching on only after
the ones before it stabilize. Whether that is a general property of ternary QAT or an
artifact of this LR schedule is exactly what a second run at a different LR would settle.

Scale drift rises monotonically alongside (1.2%–3.0% by step 300) and does **not** peak,
consistent with it being the continuous background process against which flips are the
discrete events.

## Observation 5 — training efficiency peaked at step 200 and has halved since

Codes changed per GPU-hour across the tracked sample — the ternary analogue of a
learning-rate-of-return curve, and the one number that says when to stop. Loss cannot
answer this, because loss keeps drifting down on scale drift alone.

| step | GPU-h | cumulative codes changed | **codes / GPU-hour** | codes / 1M tokens |
|---|---|---|---|---|
| 50 | 5.0 | 9,186 | 3,398 | 10,559 |
| 100 | 10.0 | 248,815 | 61,470 | 197,981 |
| 150 | 15.5 | 936,555 | 162,286 | 548,120 |
| **200** | 20.9 | 1,948,208 | **195,716** ← peak | 642,154 |
| 250 | 26.1 | 2,870,024 | 160,225 | 514,114 |
| 300 | 31.2 | 3,498,570 | 104,390 | 330,910 |
| 325 | 33.9 | 3,720,046 | 83,097 | 274,648 |

Efficiency peaked at **step 200 / 21 GPU-hours** and is now at **42% of peak**. Over the
whole run so far, 1.056% of the tracked 352M parameters have changed code.

Read against Observation 1, this is the run's actual shape: the first 25 steps did almost
nothing (warmup), steps 30–120 were spent recovering from the LR excursion, steps 120–250
were the productive core, and the tail is annealing. The remaining ~200 steps (≈20
GPU-hours) will, on this trend, produce roughly a third of the change per hour that the
peak did.

That is *expected* under cosine decay — the schedule is designed to anneal — so this is not
evidence the run is broken. It is evidence about **where the budget went**: over half the
wall-clock bought either warmup or divergence recovery. A run at a LR that doesn't diverge
should reach the same flip count in materially fewer GPU-hours, which is the concrete
argument for the 2.5e-4 control.

## Observation 6 — cost, and the memory regime

Sustained **375 s/step** (11.5 ms/token) at window 8064, MPS resident flat at 30.8 GiB. An
earlier s/step creep (356 → 372 over 100 steps) reversed on its own as allocator
fragmentation settled; swap receded from 24.6 to 23.6 GB. Full run ≈ 54 h.

Getting here took fixing four separate OOM causes, all of which produce an identical bare
`Killed: 9` with no traceback; they are catalogued in `docs/ternary_qat.md`. The one worth
repeating as a *methodological* point: the lm_head spike scaled with each window's
trainable-token count, which varies 0.05–1.00 in density, so the process died at a
**random** step and no single-window probe ever reproduced it.

## What we could not measure (and now can)

These were unrecoverable for this run — the write-up will have to say so.

| gap | why it matters | status |
|---|---|---|
| **gradient norm** (pre-clip) | the one number that explains Observation 1 | **added** |
| **per-source loss** | 5 sources, wildly different `assistant_frac` (0.08–0.79); we cannot say which data drove learning | **added** (needs `window_source`, also added to the builder) |
| flip **velocity** | had to be reconstructed by differencing; no absolute density | **added** (logged directly, plus absolute nonzero density) |
| machine-readable metrics | everything came from parsing stdout | **added** (`metrics.jsonl`) |
| signed scale drift | mean `\|Δs\|/s` hides whether scales grow or shrink | **added** |
| val resolution | 8 windows → a noisy 0.96–1.12 band; the 240→280 "uptick" is probably nothing | raise `--val-windows`; per-source val |
| rotating checkpoints | single overwritten file — could not roll back to the pre-divergence step | **added** (`--ckpt-keep`) |
| full-model −1/0/+1 census | only the 12 sampled tensors have a trajectory; the per-layer composition figure covers those | deferred — the census reads the whole 15 GB model, which is not safe to do beside a memory-tight run. Run `ternary_distribution.py census --all` after the run finishes. |
| tokens seen | throughput reported in steps, not tokens | **added** |

## Figure audit — what rendering them actually caught

The figures were rendered to PNG (`qlmanage -t`, WebKit) and inspected. Three defects were
only visible once looked at, and two of them were wrong, not merely ugly:

1. **The LR panel had a 10px plot area.** A flat `pad=60` on a 130px-tall panel left no
   drawable height: the cosine schedule rendered as a straight line and every y-tick label
   stacked on one row. Vertical padding now scales with panel height.
2. **The recruit-vs-prune scatter was not square.** It compares two like quantities against
   a 45° balance diagonal — at 900×300 that diagonal renders as a shallow slope, so
   "distance below the line" was visually distorted. Now square, and the reading changes:
   most tensors sit clearly *below* the line.
3. **The depth profile connected different tensor kinds.** A line joined layer 0 `q_proj` →
   layer 3 `v_proj` → layer 6 `gate_proj`, implying a continuous series that does not
   exist — there is one sampled tensor per layer. Replaced with lollipop stems.

A structural check (well-formed SVG, marks in bounds, label spacing) passed on all three.
Geometry validation does not catch a chart that is drawing the wrong thing.

## What the audit changed in the trainer

Two findings from the figures turned into code:

- **The divergence starts 4 steps after warmup ends** (figure 1, LR panel). Warmup is 5%
  of total steps = 26 here; the loss leaves 1.06 at step 30. `--warmup-frac` is now a flag.
- **Clipping did not prevent it, and hid it.** `clip_grad_norm_` rescales the update
  direction but still takes a full-size step along it. `GradSpikeGuard` (`--grad-spike-factor`,
  default 4.0) skips any optimizer step whose *pre-clip* norm exceeds 4× the trailing
  median, after a 20-step history has built. Skipped norms deliberately do not enter the
  history — otherwise a sustained excursion drags the median up until the guard stops
  firing, which is the exact case it exists to catch. Requires splitting
  `MasterOptimizer.clip_and_step` into `stage_grads_and_norm` + `step_staged`, since the
  decision needs the norm before the step is committed.

A third finding is **not** yet acted on: figure 4 shows flips concentrating in `q_proj`
(3.7% at layer 0, 2.9% at layer 13) while `v_proj` and `up_proj` barely move (0.13–0.33%) —
a ~30× spread. Every tensor is being trained at the same LR and costing the same memory.
Whether the low-movers are *unimportant* or merely *slow* is untested, and freezing them
would be a real efficiency win if the former. That needs an ablation, not a guess.

## Open questions for the next run

1. **Does a lower LR reach the same flip count without the excursion?** The single most
   valuable control. 2.5e-4, same corpus, same seed.
2. **Is the depth-ordered velocity peak real** or an artifact of the cosine schedule?
   A constant-LR run separates them.
3. **Which source drives the flips?** With per-source loss and `window_source`, attributable
   for the first time.
4. **Does the reorganize/densify split predict export quality?** If v_proj densification is
   what matters, a run could weight it.
5. **Does the excursion cost anything measurable at export?** Only the SWE-rebench number
   answers this.
