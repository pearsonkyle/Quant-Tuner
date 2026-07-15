# Can our calibration improve the native-ternary Ternary-Bonsai-8B?

Short answer: **post-hoc calibration cannot** (proven below), a prompt-scaffolding
probe made it cheaper but not better, and the only real lever is continued QAT
training on the logs (or an added higher-precision adapter, which changes the
2-bit artifact).

## The structural fact (why imatrix / AWQ / scale-refit are all no-ops)

Ternary-Bonsai-8B is *trained* natively ternary. We verified at the weight level:
in every 128-weight group of the shipped F16, all nonzero weights share one
magnitude (max/min = 1.0000; ~37% are exactly 0). So `w = s·c` exactly, with
`c ∈ {−1, 0, +1}` and one fp16 scale `s` per group. Q2_0 is a **lossless**
re-encode (measured KLD ≈ 0, top_p 99.99%).

Every post-hoc calibration method we have recovers **quantization rounding error** —
the gap between a higher-precision original and the low-bit grid. Here that gap is
**zero**:
- **imatrix** steers rounding decisions → no rounding freedom (weights already on grid).
- **AWQ** folds a per-channel scale and re-quantizes → perturbs QAT-optimal weights
  and forces re-rounding → strictly *adds* error.
- **Per-group scale refit (the "imatrix-for-ternary" idea):** minimizing
  `‖Wx − ŝ·c·x‖²` has its optimum at `ŝ = s` (error 0, for any activation `x`),
  because `w = s·c` exactly. Any deviation increases output error. **No-op.**

There is no higher-precision target to fit toward — the ternary weights *are* the
model. Changing the codes to beat the trained weights is not calibration; it is
training (→ QAT, below).

## #3 — prompt scaffolding (opt-in, no test-name leakage). Result: cheaper, not better.

Added `QT_SWE_EXTRA_INSTRUCTIONS=scaffold` to the openai-agents backend (default
prompt unchanged, so published Ornith/Qwythos/gemma numbers are untouched). The
guidance targets the observed failure modes: don't invent pytest flags, discover
the project's test runner, don't run library modules as scripts, and treat a
reproducing test's non-zero exit as expected signal (not an error). Same 10-instance
holdout, same sampling.

| Ternary Q2_0 | steps | tokens | tool-err | patch | pass | exits |
|:-|-:|-:|-:|-:|-:|:-|
| baseline | 15.3 | 258K | 79% | 50% | **0%** | 9 completed, 1 max_turns |
| + scaffold | 7.4 | **26K** | 62% | 30% | **0%** | 10 completed |

The scaffolding made the model disciplined — **10× fewer tokens**, no runaway loops,
fewer bad commands (79%→62%) — but the discipline meant it *did less*: patch rate
fell 50%→30% (within n=10 noise, ±~15%) and pass stayed **0%** despite *more*
empty-patch nudges (14 vs 12). The high tool-error rate was never the bottleneck —
it was a noisy metric (a reproducing test failing counts as a non-zero "error") plus
a symptom of flailing. Removing the flailing didn't add the missing capability:
producing a *correct* source fix. **Capability is the wall; prompt text can't move it.**

Shippable takeaway: if the goal were cost/latency, `scaffold` is a real 10× token
win at no resolution cost. It is not a quality lever.

## #2 — Q2_0 per-group scale refit. Verdict: **not built — proven no-op** (see structural fact).

## GPTQ — verdict: **not appropriate** (demonstrated on our own framework)

GPTQ *reconstructs* quantized weights toward a reference `W` by minimizing
`‖(W − Ŵ)X‖²` over calibration activations `X`. Two reference choices, both dead:

1. **Reference = Bonsai's own ternary weights, grid = ternary.** We ran our real
   `gptq_round_tensor(sym=True, n_bits=2, group_size=128)` — whose grid `qmax=1`,
   `scale=max|w|` is *exactly* Q2_0's `{−s, 0, +s}` — on `blk.0.ffn_gate.weight`:
   relative reconstruction error = **0.000**, `max|W − Ŵ| = 0`. GPTQ changes
   nothing, because the weights already sit exactly on that grid.
   The only way it "acts" is by targeting a *different* grid (asymmetric 2-bit,
   `sym=False`): rel error jumps to **0.333** — a lateral re-quant that is worse and
   can't be packed as Q2_0.
2. **Reference = a dense teacher (original Qwen3-8B fp16), grid = ternary.** This is
   the symmetric 3-level grid our own `gptq.py` (lines 152-157) documents as one that
   *"has only three usable levels {−s, 0, +s} … and destroys the weight
   distribution."* Post-hoc ternary rounding of dense weights is catastrophic —
   precisely why native-ternary models are QAT'd, not post-hoc quantized.

**Why every post-hoc method fails, in one sentence:** the model is not badly
*quantized* (it is a perfect, lossless quant of itself), it is badly *allocated* —
its ternary capacity was spent on general pretraining, not tool/agent use. Post-hoc
calibration reduces *quantization error*; there is none. Re-allocating capacity for
our task needs a training signal, which no calibrator has.

## The real path: continued ternary QAT on the logs (feasible; method is public)

Correction to "no published pipeline": prism's exact *data/hparams* are proprietary,
but the *method* is **published — BitNet b1.58** (Microsoft, 2024): straight-through
estimator + absmean ternarization, a ~20-line PyTorch module. So this is not
inventing a secret recipe; it is applying a known one as a short *continued* fine-tune.

Base is ready: `prism-ml/Ternary-Bonsai-8B-unpacked` is a **plain, trainable
`Qwen3ForCausalLM`** (36 layers, hidden 4096, GQA 32/8; ternary weights stored as
fp16, no quant_config) — standard HF/PyTorch training.

Sketch:
1. Wrap each linear's forward with STE ternarization: `s = mean(|W|)` per group,
   `Ŵ = s · clip(round(W/s), −1, 1)`; forward uses `Ŵ`, backward flows to the fp16
   latent `W` (STE). Initializing latents from the shipped fp16 reproduces the
   current model exactly at step 0.
2. Continue-train on `corpus.cal.txt` (our tool-call logs) with LM cross-entropy,
   low LR, a few hundred–thousand steps (fine-tune, not from scratch).
3. Re-ternarize final latents → pack to Q2_0 (prism format) → bench + SWE-rebench.

### iter-2 RESULT (exp-058): masked-loss QAT changed behavior, did NOT improve outcomes

Ran it end to end — masked-loss (tool-call tokens only) + rendered tool schemas +
turn-aware 4096 windows, fp32 last-18, 261 steps (~0.5 epoch). Masked training loss
fell **2.26 → ~1.0**. Exported to Q2_0 (2.03 GiB), tool-calling verified. Full
10-instance SWE-rebench vs the untrained baseline:

| SWE-rebench (n=10) | Baseline Ternary Q2_0 | **iter-2 QAT** | Δ |
|:-|-:|-:|-:|
| Pass rate | 0% | **0%** | — |
| Patch rate | 50% (5/10) | 40% (4/10) | −1 (within n=10 noise) |
| Tool-error rate | 79.1% | **67.8%** | **−11 pp** |
| Mean tool steps | 15.3 | **95.4** | **6.2×** |
| Mean tokens | 258K | 556K | 2.2× |

**Verdict: the QAT taught tool-call *style*, not issue-resolution *capability*.**
- ✅ Tool-error rate dropped 11 pp — the masked training genuinely produced cleaner,
  better-formed tool calls (the signal was real and correctly targeted).
- ✅ Engagement rose sharply (6× steps, 2× tokens) — training on the agentic logs
  made the model far more persistent about calling tools.
- ❌ **Patch rate did not improve** (40 vs 50%, statistically indistinguishable; iter-2
  patched a strict *subset* — gained NONE of the 5 instances the baseline failed, and
  lost the hardest one it had, durable-python-494 / 14 F2P tests). The extra
  engagement did not convert to diffs.
- ❌ **Pass rate stayed 0%** — no instance resolved, same as every 2-bit model.

**Follow-up — the looping cause + a repetition-penalty A/B.** The 6× step count was
pathological looping: the QAT model repeated whole commands (up to **553/556 duplicate
tool calls** on one instance, several hitting max_turns / crashing the server). The
masked training on long agentic sessions taught *persistence* but not *when to stop*,
and at 2-bit that degraded into repetition. A stronger server-side penalty
(`--repeat-penalty 1.25 --repeat-last-n 1024 --frequency-penalty 0.4`, now env-tunable)
**killed the loops** (mean steps 95→2.9, tokens 556K→11K, dup_cmds→~0) — but
**over-corrected**: the model now quits after ~3 calls and patches even fewer (30%).

| SWE-rebench (n=10) | patch | tool-err | mean steps | mean tok |
|:-|-:|-:|-:|-:|
| baseline (no QAT) | **50%** | 79% | 15.3 | 258K |
| iter-2 QAT (pen 1.1) | 40% | 68% | 95.4 (loops) | 556K |
| iter-2 QAT (pen 1.25+freq) | 30% | 76% | 2.9 (gives up) | 11K |

So there's a Goldilocks band (an intermediate penalty might land ~40%), but **no
setting beat the plain baseline's 50% patch, and pass stayed 0% everywhere.**

The 2-bit ternary **resolution floor is a capability wall a light (0.5-epoch, last-18)
fine-tune does not break** — consistent with the whole study's finding that ~zero
issues resolve at 2 bits regardless of model or calibration. The tool-error drop
shows QAT *is* doing something useful; scaling it (full epochs, all layers via the
fp32-master trick, or a stronger base) is the next lever, but pass@2-bit may be
fundamentally capped. What's proven: a **working, reproducible Metal ternary-QAT
pipeline** that measurably moves the model — the infrastructure for that next attempt.

Cost/risk: full 8B QAT needs a GPU with room for fp16 latents + AdamW states +
activations (a CUDA box, or a memory-frugal LoRA-on-latent / partial-layer variant on
Metal). This is real training infra quant-tuner doesn't have yet — the actual build.

### exp-057: the QAT pipeline is BUILT and VALIDATED on Metal (M4 Max, 128 GB)

`src/quant_tuner/qat/ternary.py` (per-group TWN + STE) + three scripts
(`exp057_qat_{step0,train,export}.py`). All green:

- **Step-0 reproduction is EXACT** (fp32): wrapping all 252 linears with the STE
  ternarizer changes the logits by `0.0000e+00` (100% top-1). The fine-tune
  provably starts from the real model — TWN recovers each group's `s` exactly.
- **Training runs on MPS**: fp32 8B, last-4-layers trainable, gradient
  checkpointing → **36.6 GiB** peak (of 128), ~20 s/step, loss falling
  (2.34 → 2.10 in a 20-step smoke run). Frozen layers carry no optimizer state.
- **Export round-trips**: trained latents → ternarize every linear → F16 GGUF →
  prism `llama-quantize` **Q2_0** (supported, type 41). The result runs on the
  prism fork and generates coherent code; the trained layer (blk.35) changed,
  the frozen layer (blk.0) is byte-identical.

So Metal is sufficient — no CUDA needed. Remaining for a *real* run (not just a
smoke test): (a) a tool-log-weighted training corpus (the cal corpus is wiki-heavy);
(b) more trainable layers (128 GB fits all 36) + more steps; (c) match the original
2.03 GiB packing (`--output-tensor-type`/`--token-embedding-type Q2_0`; our default
left embed/output at higher precision → 2.53 GiB); then bench + SWE-rebench vs the
0%-pass baseline.

## What would actually improve it (the real levers)

1. **Continued QAT fine-tune on the tool-call logs** — the correct analog to
   "calibrate on our distribution" for a QAT model. Continue-train the trainable
   checkpoint (`prism-ml/Ternary-Bonsai-8B-unpacked`) on the logs with a
   straight-through estimator so weights stay ternary. Real training (gradients +
   STE), heavier than our post-hoc pipeline, but the only thing that raises
   capability at 2 bits. This is what imatrix/AWQ *approximate* for post-hoc quants;
   for a native-ternary model you do the real thing.
2. **A small higher-precision adapter (fp16 low-rank correction) on top of the
   ternary base**, fit on the logs. Much cheaper than full QAT and can lift
   capability — but it *adds* precision/params, so the artifact is no longer pure
   2-bit Q2_0. A different point on the size/quality curve, offered honestly.

The one-line lesson, which is also the "KLD isn't everything" thesis in its purest
form: a model can be a **perfect** quantization of itself (KLD ≈ 0) and still resolve
**0%** of real issues — and no amount of *quantization* calibration changes that,
because there was never any quantization loss to recover.

### Penalty sweep (4-way) + the layer-selection pivot

Full A/B of sampling penalty on the iter-2 (last-18) QAT model:

| run | patch | tool-err | mean steps | dup/calls |
|:-|-:|-:|-:|-:|
| baseline (no QAT) | **50%** | 79% | 15.3 | 116/153 |
| QAT pen 1.1 | 40% | 68% | 95.4 | 912/953 (loops) |
| QAT pen 1.25 + freq 0.4 | 30% | 76% | 2.9 | 3/29 (quits) |
| QAT pen 1.15 / last-n 512 | 30% | **61%** | 13.7 | 56/137 (healthy) |

Penalty 1.15 restored *healthy behavior* (steps ≈ baseline, lowest tool-err) but
patch rate did NOT recover — **no penalty beats the plain baseline's 50%**. The loop
was a symptom; the last-18 QAT genuinely lowered patch capability. Combined with the
**layer-importance probe** (grad signal on the masked tool-call loss: blk.0 dominates,
early layers 0-14 carry the signal, layers 18-31 — exactly what iter-2 trained — are
the WEAKEST, 32/34/35 spike), the likely cause is that iter-2 trained the wrong half.
**iter-3** retrains the grad-influential set (0-14, 32, 34, 35) at the same budget to
test whether smart layer selection beats naive last-18.

### iter-3: train the grad-INFLUENTIAL layers (0-14,32,34,35) + error-feedback harness

Retrained the layers the gradient probe ranked highest (early layers 0-14 dominate;
naive last-18 trained the weakest, 18-31), same masked 4096 corpus + budget. Also made
the openai-agents backend feed BadRequest-style errors back and restart instead of
aborting the instance.

| SWE-rebench (n=10) | pass | patch | tool-err | mean steps | dup/calls | exits |
|:-|-:|-:|-:|-:|-:|:-|
| baseline (no QAT) | 0% | **50%** | 79% | 15.3 | 116/153 | — |
| iter-2 (last-18) | 0% | 40% | 68% | 95.4 | 912/953 | looped / crashed |
| iter-3 (influential) + errfix | 0% | 40% | 60% | 22.6 | 184/226 | **10/10 completed** |

**What the right layers + error-feedback fixed:** the LOOPING (steps 95->22.6,
dup-commands 912->184) and the instance CRASHES (10/10 completed, no BadRequest
aborts). Training the influential early layers is clearly better-behaved than the naive
last-18. **What it did NOT fix:** patch rate (still 40%, below baseline's 50%) and pass
rate (still 0%). Layer selection + harness robustness recovered iter-2's behavior
regressions but did not push capability past the untrained baseline. The 2-bit
resolution wall holds; a light half-epoch fine-tune (any layer set, any sampling) does
not beat it. The remaining levers are scale (full epochs / all layers via fp32-master /
more data) or accepting the wall.
