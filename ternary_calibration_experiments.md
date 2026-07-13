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
