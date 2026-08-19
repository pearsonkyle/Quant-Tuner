# Fine-Tuning a Natively-Ternary LLM Without Breaking It

**Continued QAT on Ternary-Bonsai-8B for agentic tool use — condensed findings (exp-058)**

*Companion docs: `ternary_qat_curriculum.md` (full experimental arc), `ternary_qat_reproduce.md`
(runbook), `ternary_qat.md` (pipeline guide). Every number below is measured; sources in the
curriculum doc.*

## Abstract

We fine-tune a natively-ternary 8B model (weights w = s·c, c ∈ {−1, 0, +1}) on agentic coding
logs with straight-through-estimator QAT. Naive masked-CE fine-tuning reliably destroys the
model's *termination policy*: P(stop token | completed sentence) rises from the shipped model's
0.0092 to ~0.95 while validation loss stays flat, yielding agents that quit mid-task or loop.
We show why standard losses are blind to this failure and present a recipe that completed a
full schedule with termination intact: offline top-K distillation with a tail-bucket KL and
forced-stop support, a one-sided stop anchor with per-side margins, probe-context steering,
tight gradient clipping, scale-aware learning rates, and in-training termination probes with
patience-gated aborts. The result is the first checkpoint in this series to finish an agentic
episode cleanly.

## 1. Setup

Student: Ternary-Bonsai-8B (Qwen3-family, 36 layers). Ternarization is per-group TWN, group
128, threshold δ = 0.7·mean|w|. The shipped "F16" is a lossless container of the ternary
weights, so fine-tuning starts **exactly on the quantization grid** — zero quantization error
at step 0. This is a luxury post-hoc quantization lacks, and the reason our measured constants
do not transfer to ternarizing a dense model from scratch (measured on gemma-4: rel-Frobenius
distance to the grid ≈ 0.40 on every tensor, for stock *and* Q4-QAT checkpoints alike).

Training: all 36 layers (6.95B trainable), **fp32 latents** (bf16 underflows δ — zero code
flips, i.e. no learning), Adafactor (its factored state is 0.03% of params; an 8-bit optimizer
is a no-op here — memory is params + grads + activations), 32768-token windows, loss masked to
assistant/tool-call tokens **plus the terminating stop token**. Corpus: real agent sessions
(median 32k tokens) plus SWE trajectories and broad SFT, packed per-source.

## 2. The failure mode: termination collapse

Every early run broke the stop decision — premature stopping (quits after one step) or
non-termination (verbatim loops). Four measurements localize the cause:

- **Masked CE cannot see it.** One run's validation was flat for 225 steps while
  P(stop | sentence end) went to 0.97. Stop decisions are ~1 per 176 supervised tokens; their
  CE contribution is noise.
- **Loss weighting does not fix it.** Stop-token CE weights 1.0 and 6.0 reach the same
  collapsed endpoint (6× change moved the diagnostic by 0.02).
- **The corpus alone does not cause it.** The corpus teaches at most P(stop) ≈ 0.055 in the
  probe's own situation; trained models reach 0.95. (Two real ingestion defects — one
  assistant turn split into prose + tool-call messages, and 2,155 empty assistant messages
  whose only supervised token was the stop token — were found and fixed, but the fix alone did
  not prevent collapse.)
- **It is not kernels, drivers, or math.** A dense control (identical config, quantizer off)
  reproduces a slow collapse at low lr; at the ternary-required lr 5e-4 the *dense* model
  diverges in 10 steps (loss 15, grad-norm 47). Reproduced on both MPS and CUDA; step-0
  weights reproduce the shipped model exactly; torch and GGUF probes agree. Decomposition:
  (1) a slow objective-driven drift toward the control contexts, (2) oscillatory waves from
  objective × fixed data order, (3) the quantizer acting as a low-pass filter over
  divergently-hot dynamics — what leaks through appears as threshold-crossing bursts.

## 3. What worked

Each rung was added after a measured failure of the previous one (§4):

1. **In-training stop probe** (0.7 s, every 25 steps): P(stop) at a fixed *diagnostic* context
   (mid-prose — stopping is wrong) and *control* context (after a tool call — stopping is
   right). The single highest-value instrument; validation loss is flat through the collapse,
   the probe is not.
2. **Offline top-K distillation with a tail-bucket KL.** Teacher top-64 logprobs precomputed
   once to a CPU table (~0.38 KB/position; no teacher in GPU memory, so it composes with
   full-model training). The KL runs over K+1 buckets: the stored support at true
   probabilities plus a tail bucket for everything else. The obvious alternative —
   renormalizing both sides over the support — is **blind to student mass placed outside the
   teacher's top-K**, and the stop token is outside the teacher's top-64 at **98.2%** of
   positions: exactly where the collapse lives.
3. **Forced-stop support** (`--include-ids <stop_id>` at precompute): the stop token enters
   every stored row at its true teacher logprob, making the KL an exact per-position
   constraint on P(stop) rather than a cap on total tail mass.
4. **One-sided stop anchor with per-side margins** (hinge on the log-P(stop) gap to the
   teacher; margin 1.0 nat at continue-positions, 0.1 nat at stop-positions). Direction is
   chosen per position type: push *down* only where stopping is wrong, *up* only where it is
   right. Weight 0.2.
5. **Termination steering** (weight 0.1): eight probe-*family* contexts forwarded every step —
   CE toward stop after tool calls, a one-sided hinge above a cap (log 0.02) mid-prose. The
   probe's own texts are asserted held out, so the probe stays an honest measurement
   (Goodhart guard).
6. **Clip-norm 0.25** (from 1.0): damps the wave amplitude of §2's oscillations without
   suppressing code flips.
7. **Scale-aware per-tensor lr** (∝ median TWN group scale, clamped [0.5, 2.0]): a code flip
   requires moving a latent ~δ ∝ s, but Adafactor's normalized step is scale-blind, so
   low-scale tensors were flip-starved.
8. **Dual abort guards with patience 2**: abort on diagnostic > 0.09 *or* control < 0.95, but
   only on consecutive violations; recovery resets the strike. A single-reading abort killed a
   run at an oscillation trough that had already recovered once.

**Outcome** (recipe = all eight, lr 5e-4, ~1 epoch, 613 steps): perfect probe record
(diagnostic 0.0000 / control 1.0000 on all 24 probes), code flips 1.07–2.37% on every tracked
tensor, validation loss equal to the dense control's endpoint, guards never fired. Exported to
2-bit GGUF: probe survives serving numerics; in-distribution median P(stop) at real stop
positions 0.995; the agent completes a 10-step episode and self-terminates.

## 4. What didn't work

- **Stop-token CE weighting** (any weight — §2).
- **Support-renormalized KL** (blind exactly at the stop token — §3.2).
- **Symmetric anchor**: the 176:1 continue:stop imbalance makes a direction-free penalty
  net-downward — diagnostic pinned at 0.0000 while control collapsed to 0.70.
- **A single margin**: 1 nat below P = 0.99999 is P = 0.37, so the anchor read "satisfied"
  through the entire control-side collapse.
- **Single-reading aborts**: kill recoverable oscillation troughs.
- **lr 3e-4**: loss falls but ~0% code flips — scale drift only; the model is not learning.
  Read **flip telemetry, not loss**. (Conversely ~8 epochs at 5e-4 memorizes.)
- **bf16 latents** (underflow δ) and, on Metal at full-model scale, bf16 compute (the fp32
  master copy stacks on top: 54.5 vs 31 GiB).
- **Single-window memory probes** for choosing the window size: they omit the optimizer step
  and run before swap builds; trust only measured s/step in the real loop.

## 5. Remaining gap and practicalities

With termination fixed, the frontier failure is **verbatim action repetition** — the agent
re-issues the identical command (up to 49×) after unhelpful results: a state-tracking failure,
not a termination one (it stops correctly after every call). Serving-side repetition penalties
(`--repeat-penalty 1.3 --repeat-last-n 2048 --presence-penalty 0.8`) eliminate it; a
training-side counterpart (one-sided hinge on the mean per-token logprob of re-issuing the
previous command in synthetic command→failure→retry contexts) is implemented and under test.
Capability at 2 bits — actually resolving hard issues — remains open; behavior no longer is.

Evaluate on **three instruments together**, each blind somewhere: the in-training probe (can't
see multi-turn loops), P(stop) at real corpus stop positions (catches a bimodal weak tail the
probe's medians hide — one collapsed run had a healthy median and p10 = 0.08), and a real
agentic episode (can't distinguish early stopping from incapacity). The shipped model itself
proves the gap: textbook probe, looping trajectory.
