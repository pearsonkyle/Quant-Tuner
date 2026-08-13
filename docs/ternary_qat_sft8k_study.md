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

Figure: `out/exp-058/telemetry/ternary.html` (per-layer −1/0/+1 composition + zero-fraction
trajectory), regenerated with `scripts/ternary_distribution.py plot`.

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

## Observation 5 — cost, and the memory regime

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
