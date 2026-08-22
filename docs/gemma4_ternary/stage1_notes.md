# gemma-4-E4B stage 1 — go/no-go notes

Written **before** launch. Criteria are fixed here so the read afterwards is not a
negotiation with the numbers.

## The question

> Does QAT recover a stage's ternarization damage before the next stage compounds on it?

With no training the cumulative curve (`layer_damage.json`, held-out KLD vs the dense
model, same 3x2048-token probe throughout) doubles every 6 layers and the individual
layers are **3.77x superadditive**:

| ternarized layers | KLD(dense ‖ cand) | PPL | top-1 agree |
|---|---|---|---|
| 6  | 0.105  | 4.10    | — |
| 12 | 0.269  | 4.71    | — |
| 18 | 0.610  | 6.18    | — |
| 24 | 1.222  | 10.67   | — |
| 30 | 2.171  | 25.42   | — |
| 36 | 5.288  | 541.77  | — |
| 42 | 10.666 | 102,989 | 4.7% |

Stage 1 is the first 6 of `layer_damage.json["layer_order"]` — `0,1,2,3,7,8`.
`down_proj` stays dense in every stage (its solo KLD is 1.199, 3.4x the next-worst kind).

**The stage's own untrained baseline is 0.0762, not the table's 0.1047.** That row
ternarized every linear in the six layers; a stage holds `down_proj` dense, and skipping
the single most damaging kind removes 27% of the damage before training does anything.
Measured with `scripts/gemma4_stage_damage.py` (same probe, same process, dense
self-check 0.0e+00): `kld=0.0762 top1=0.916 ppl=4.05` against the dense model's
`ppl=4.108`. Note ppl *falls* slightly while KLD rises — perplexity on this probe cannot
see this damage at all, which is why the criteria below are KLD.

## Hypothesis

Training the stage recovers most of its own damage, so that each stage starts from
near-parity rather than from the previous stage's residue. Concretely: the compounding
above is a property of the *untrained* composition, and the doubling per stage is not a
law but the absence of any correction between stages.

## Criteria, fixed in advance

Measured with `scripts/gemma4_stage_damage.py` against the same held-out probe, so the
number is directly comparable to the table above.

Stated as **recovered fraction** `(untrained - trained) / untrained`, not an absolute
KLD, so the same criterion carries to stage 2 without being re-derived (each stage has
its own baseline, and stage 2's will be larger).

| outcome | recovered | KLD at the 0.0762 baseline | read |
|---|---|---|---|
| **GO** | >= 70% | <= 0.023 | a 7-stage schedule is worth running |
| **marginal** | 30-70% | 0.023-0.053 | recovery is real but partial - one diagnostic iteration (lr A/B, longer stage, wider dense set) before committing |
| **NO-GO** | < 30% | >= 0.053 | training barely moves it; the schedule buys nothing over all-at-once and the honest verdict is that fully-ternary E4B is out of reach at this budget |

Secondary gates, each of which can independently fail the stage:

- **Termination.** `stop_baseline.json` is the reference, not Bonsai's:
  diagnostic `sentence_period` **0.00274**, control `answer_after_tool` **0.0703**.
  Abort at diagnostic > 0.03 or control < 0.01, patience 2. gemma's control headroom is
  only ~25x (Qwen had ~10^4), so a control move is checked against a generated
  trajectory before the probe is believed in either direction.
- **Code flips.** A ternary model learns only by flipping codes. A run whose loss falls
  with ~0% flips has drifted scales and learned nothing — read the flip panel next to
  the damage number, never the loss alone.
- **Val trend.** Masked CE on `corpus_sft_gemma4_val_32768.pt` (86 windows, fingerprint
  `16177b9a361cbdd7`), disjoint from train by session group.

## The lr A/B, pre-registered

`lr 5e-4` is Bonsai's measured sweet spot and is only a first guess here, because the
two situations are not the same one. Bonsai's weights START on the ternary grid, so the
only question is whether the lr is large enough to flip codes at all (measured: 3e-4
flips ~0% and drifts scales while the loss falls). gemma's weights start OFF the grid,
so step 0 is a large perturbation and there is real gradient signal from the outset; the
risk shifts from "too small to move anything" toward "large enough to break
termination".

Three 60-step arms (`EPOCHS=0.0922` at `GRAD_ACCUM=1` over 651 windows), identical but
for lr: **2e-4 / 5e-4 / 1e-3**. Accum 1 at a 32768 window follows the Bonsai full-run
precedent and gives 651 steps per epoch, so a 60-step arm is a real read and
`--probe-every 25` samples termination three times inside it. At accum 4 one step is
131k tokens, a 60-step arm sees a third of the corpus in 60 blunt updates, and the probe
fires only twice. Each is read on four things, and no single one decides:

1. **flip %** — near-zero means the arm learned nothing regardless of its loss.
2. **damage** (`gemma4_stage_damage.py`, the go/no-go metric) at 60 steps.
3. **stop probe** vs 0.00274 / 0.0703.
4. **val masked-CE** trend.

Pick the largest lr that is still flipping codes and holding termination, then run the
full stage at `EPOCHS=2.0` (326 steps). If no arm recovers meaningfully by 60 steps that
is itself informative, but it is NOT the NO-GO verdict — 60 steps is a sixth of the
stage, and the verdict is read at the end of a full stage.

## Known confound, declared up front

The teacher is **`google/gemma-4-31B-it`** — a different, larger model, not the
student's own dense self. So "KLD vs dense E4B" now mixes two effects: ternarization
damage (down is recovery) and the student legitimately moving toward a better
distribution (up is not necessarily damage). The GO/NO-GO table above is written on the
assumption that recovery dominates at this stage, which is checkable: if the stage lands
marginal or worse, the **self-KD control arm** (teacher = the dense E4B itself, table
`e4b_self_topk64_fs106.pt`) re-runs the identical stage with the confound removed. Both
tables are precomputed; the control arm is not a new experiment, it is one command.

## Log

- `2026-08-21` — teacher gate passed: gemma-4-31B-it and the E4B student agree on all
  **262,144** ids (this is what refused the obvious Qwen teacher in the Bonsai arc).
- `2026-08-21` — KD precompute OOM'd on the first try at 94 GiB. Not the weights (61
  GiB): a **32 GiB KV cache** built by a forward-only pass. gemma-4-31B has 16 KV heads
  at head_dim 256, 4x Qwen3-32B's KV width, so the same latent bug that cost 8.6 GiB
  there is fatal here. Fixed with `use_cache=False`.
- `2026-08-21` - corrected the stage-1 baseline to **0.0762** (see above) and restated
  the criteria as a recovered fraction. `scripts/gemma4_stage_damage.py` is the harness;
  it wraps the model with the trainer's own `wrap_model` rather than re-deriving the
  ternarization, so what it measures is what training deploys.
- `2026-08-21` - the report's termination panel hard-coded Qwen's control point
  (`after_tool_call`) and Qwen's reference values. On gemma that point reads **0.00004**
  on the shipped model, so the panel would have drawn the most-broken-looking line as
  the healthy control. It now reads `PROBE_SPECS`, detected from the points present in
  the run's own log; Qwen's published reference line is pinned unchanged.
- `2026-08-21` - CPU trainer smoke (2 layers, 2 windows x 2048) confirms the gemma path
  end to end: dialect detected (7 probe points, stop id 106), 16 latents wrapped with
  `down_proj` held dense, group-scale lr, adafactor, flip telemetry. **Step 1 reads
  `loss=7.6378 gnorm=48.10`.** That gnorm is the thing to watch: `--clip-norm 0.25` is
  Bonsai's number, and Bonsai starts exactly ON the ternary grid, so its step-0 gradient
  is an ordinary fine-tuning gradient. gemma starts OFF the grid, so step 0 carries the
  whole ternarization perturbation and the clip is rescaling by ~190x. If the A/B arms
  come back flip-starved, clip is the second knob to vary, not lr alone.
- `2026-08-21` - the report's step-0 census defaulted to Bonsai's
  (`out/exp-058/census_step0.csv`), whose tensor names are `model.layers.N....` against
  gemma's `model.language_model.layers.N....` -- they can never join, so the
  distribution-shift panel would have rendered empty rather than wrong. Generated
  `out/gemma4-ternary/census_step0.csv`; pass it as `CENSUS=` to the report watcher.
  `ternary_distribution.py census` needed a fix to read a single-file checkpoint (gemma
  ships 15.9 GB as one `model.safetensors`, with no index). Its zero-fraction reads
  **42%**, matching the Gaussian value from the weight-space scan -- the same null
  result seen from a third angle. Bonsai's is 34.5%: a natively-ternary model has a
  genuinely denser code distribution than TWN-on-Gaussian produces.
- `2026-08-21` - the CPU smoke ran to completion (2 steps, checkpoint written) and the
  flip telemetry carries the first encouraging signal: **1.39-1.70% of codes flipped in
  two steps at lr 5e-4**, on every tracked tensor. That is the opposite of the Bonsai
  failure mode (3e-4 flips ~0% and the loss falls on scale drift alone), and the reason
  is structural: Bonsai's weights start ON the grid, so a flip needs a real move, while
  gemma's start off it with a large fraction sitting near the TWN threshold.
  The decomposition says exactly that -- every flip is `0<->±` (0->± 37,782, ±->0
  43,875 on one tensor) and `±->∓` is **0** across the board. Threshold crossings, not
  sign reversals: `Delta = 0.7*mean(|W|)` moves as the weights train, and a true sign
  flip needs a weight to cross zero, which is a far larger move. Density 58.8 -> 58.7%,
  scale drift ~1.1%.
  Caveat: two steps at 2048 tokens is not evidence about the full stage, only that the
  lever is connected.
- `2026-08-21` - the smoke also caught a bug in the damage harness. A stage's checkpoint
  holds **18** tensors for a 2-layer stage, not 16: the ternary latents plus the
  `--dense-kind down_proj` weights, which are trainable and DID train. Loading only the
  latents would measure a model that was never trained, and the strict matcher would
  have raised on the two extras. Both kinds are loaded now, still refusing a partial
  match in either direction.

## Size accounting for `--dense-kind down_proj` (raised, not resolved)

Measured from the checkpoint's own tensor shapes:

| kind | params | share of decoder linears |
|---|---|---|
| `mlp.down_proj` | 1.101 B | **28.1%** |
| `mlp.gate_proj` | 1.101 B | 28.1% |
| `mlp.up_proj` | 1.101 B | 28.1% |
| `self_attn.{q,o}_proj` | 0.257 B each | 6.6% each |
| `self_attn.{k,v}_proj` | 0.037 B each | 0.9% each |
| `per_layer_input_gate` | 0.028 B | 0.7% |

3.918 B decoder linears of a 7.941 B model. Holding `down_proj` dense in **bf16** is
therefore not a small carve-out:

    ternary   2.817 B @ 2.125 bpw = 0.748 GB
    down_proj 1.101 B @ 16 bpw    = 2.202 GB   <- 2.9x the ternary trunk
    down_proj 1.101 B @ 4.5 bpw   = 0.619 GB   (Q4_0)

A "ternary" E4B whose `down_proj` is bf16 is a model dominated by the one kind we
declined to ternarize.

**The feasibility doc already settled the packaging** (`docs/gemma4_ternary_feasibility.md`,
size-economics section) and settled it better, because it accounts for embeddings and
towers too: `down_proj` at Q4_0 costs 0.33 GB and moves the artifact from 0.79x to 0.86x
of Google's Q4_0. The table above only re-derives the trunk half of that from the
checkpoint's own shapes.

What is NOT yet settled is the training question it implies: if `down_proj` ships at
4 bits, it should be **quantization-aware at 4 bits during QAT**, not held in bf16 and
quantized afterwards -- otherwise every stage trains its dense neighbours against a
`down_proj` more precise than the one that will run. The current `--dense-kind` mechanism
has no third state for "quantize this one, but not to ternary". Out of scope for stage 1,
which asks only whether damage recovers; on the record before the schedule is committed.
- `2026-08-21` - the damage harness validated against a real checkpoint, and its first
  reading is a useful warning. The 2-step CPU smoke went `untrained kld=0.0180` ->
  `trained kld=0.0511`, i.e. **recovered -184%**: two steps made the damage three times
  worse while flipping 1.4-1.7% of codes. The configuration is meaningless as a training
  result (2 steps at full lr on a 2-window, 2048-token corpus is noise injection, and
  `--warmup-frac 0.05` of 2 steps is no warmup at all). What it establishes is that
  **flip % is not a health metric on its own** - codes moved, and they moved the wrong
  way. Only the damage number can tell those apart, which is why the A/B is read on four
  columns with no single one deciding. It also shows the harness is correctly signed and
  sensitive at the 0.01 scale the criteria live at.
- `2026-08-21` - set `GRAD_ACCUM=1` (was 4) for the same reason the Bonsai full run uses
  it at this window size. Full stage is 1 epoch = 651 steps; if damage is still falling
  at the end, extend rather than starting long.

## Accumulation: why 1 now, and what would raise it

Feasibility first -- accum 1 gives 651 steps/epoch, so a 60-step arm is a real read and
the probe samples termination three times inside it. ~9.4k supervised tokens per update
at 32k context is not a small batch; it is what the Bonsai full run used at this window.

Two measured reasons accumulation will matter for a real full run, recorded now so the
decision is not re-derived from scratch:

1. **The effective batch varies 20x step to step.** Supervised tokens per window run
   1,657 / 9,407 (mean) / 32,011, because window density ranges 0.05-1.00. At accum 1
   one step is one window, so consecutive updates are estimated from wildly different
   amounts of signal. Accumulation smooths that heterogeneity, not just the variance.
2. **Clipping has removed the natural damping.** `gnorm=48` against `--clip-norm 0.25`
   is a ~190x rescale, so every step is a FIXED-LENGTH step along a noisy direction.
   Normally a noisy gradient is also a small one and the step self-damps; clipping
   removes that, which makes the direction estimate the only thing left to improve --
   exactly what accumulation improves.

**Sequencing follows from the lr coupling.** An lr tuned at accum 1 does not transfer to
accum 2. So the arms and the FIRST full stage both run at accum 1, and the tuned lr is
the lr that runs. Raise accum only on observed instability (loss spikes, probe
oscillation, flip direction reversing) and re-check lr with it rather than carrying it
over.
- `2026-08-21` - de-risked the teacher-probe step (runbook 7b) on CPU, using E4B as its
  own teacher, so the 31B run is a validated 2-minute job when the card frees. It also
  cross-validates the instrument: `teacher_stop_probe.py` and `measure_stop_baseline.py`
  are separate scripts and agree on all seven points to **9e-06 relative**
  (`sentence_period` 0.00274438 vs 0.00274440, `answer_after_tool` 0.07031588 vs
  0.07031582). That is worth knowing rather than assuming, because the report draws the
  TEACHER's values as dotted asymptotes against the STUDENT's in-training series -- if
  the two paths disagreed, the asymptote would be wrong exactly where the panel is meant
  to be trusted.

## The KD table, and the number that actually matters about it

`2026-08-21 22:09` — 651/651 windows, **6,124,496 positions** x top-64, 2.3 GB, 6.24 h.
Verification passed: fingerprint `0c70d992882d29a7`, every window present, forced id
`[106]`, support coverage **0.9993**.

But coverage says the stored top-K captured the teacher's mass; it says nothing about
whether the teacher is RIGHT where it matters. So, measured on the corpus the student
actually trains on (`scripts/kd_stop_signal.py`):

| teacher P(stop) | n | mean | p25 | median | p75 | p95 |
|---|---|---|---|---|---|---|
| at a real stop target | 6,300 | **0.4771** | 0.0018 | 0.3871 | 0.9903 | 0.9989 |
| everywhere else | 6,118,196 | **0.000117** | 0.0 | 0.0 | 0.0 | 0.0 |

**A 4,090x discriminative ratio on exactly the decision this pipeline keeps breaking.**
The signal is in the table; whether the student learns it is now a training question, not
a data question.

**The teacher probe reads ~0 at every point, and that is not a contradiction.** The fixed
probe scores one position; at a stop target the teacher frequently prefers a newline
FIRST (`'\n\n':0.815` ahead of `'<turn|>':0.002` on one window, `'<turn|>':0.790` on
another) — the same preference the shipped E4B shows, which is why gemma has no sharp
stop point and why its control has only ~25x of headroom. The bimodality at stop targets
(p25 0.0018, p75 0.9903) is that split. Do not feed the near-zero probe values to the
report as asymptotes: they would read as "drive P(stop) to zero", which is the failure
mode, not the target. The corpus-conditioned table above is what KD actually transfers.

## Stage-1 arms, live

`2026-08-21 22:11` — arm 1 (lr 2e-4) training. **`mem=30.3/61.9 GiB`** on a 95 GiB card,
**46.8 s/step** — the last untested risk, and it fits with room. 60 steps ≈ 47 min per
arm. `loss=1.4229 kl=1.1170 an=0.5153 gnorm=31.63` at step 1.

## Arm 1 (lr 2e-4): PROBE-ABORT at step 50, and the attribution

The guard fired on the **control**, not the diagnostic:

    PROBE-ABORT: answer_after_tool=0.0043 < --probe-abort-control 0.01 for 2 consecutive
    probes — the model is losing the ability to STOP where stopping is right

i.e. the LOOP failure, not early termination. The diagnostic stayed at 0.0055 against a
0.03 threshold.

`gemma4_stage_damage.py --probe` decomposes it, which the in-training series cannot do
(its first reading is already 25 steps deep):

| | KLD vs dense | diagnostic | control |
|---|---|---|---|
| dense | 0.0000 | 0.0027 | 0.0703 |
| untrained ternary (6 layers) | 0.0762 | 0.0097 | **0.0734** |
| trained (50 steps, lr 2e-4) | **0.2724** | 0.0055 | **0.0043** |

**Ternarization is exonerated.** Six ternarized layers leave the control at 0.0734,
slightly ABOVE dense. Termination survives ternarization untouched.

**Training did both harms.** The control fell 17x and the damage got 3.6x WORSE
(recovered **-257.7%**). Not a failure to recover — an active move away.

### The mechanism is the teacher, and this is the declared confound firing

The 31B teacher's own probe reads **0.0000 at every point**, including
`answer_after_tool` where E4B reads 0.0703. KD pulls the student toward the teacher's
distribution, and the student's control went 0.0734 -> 0.0043, heading for the teacher's
0.000. **The stop anchor cannot defend against this**: its target is the teacher's
per-position P(stop) from the same table, so it anchors to the same wrong policy.

The KLD rise has the same cause and was declared before launch: KLD is measured against
dense E4B while KD pulls toward a DIFFERENT model, so "damage" and "moving toward the
teacher" are summed in one number. The pre-registered remedy is the same for both — the
**self-KD control arm**, teacher = the dense E4B itself. Then the teacher's termination
policy IS the target policy and the KLD metric is unconfounded. Two problems, one fix.

Note this does NOT contradict the 4,090x stop-signal ratio in the table: the teacher does
stop at the corpus's real stop targets (P=0.477). The probe positions are synthetic ones
where E4B stops and the 31B, rendered through E4B's template, does not.

## Arm 2 (lr 5e-4): the same failure, faster — so it is the teacher, not the lr

    step 25  control 0.0000 (arm 1 read 0.0041 here)  diagnostic 0.0004
    step 50  control 0.0000                            diagnostic 0.0005   -> PROBE-ABORT

The **diagnostic fell BELOW vanilla** (0.0004 vs 0.002744). The model is not shifting
where it stops, it is losing P(stop) *everywhere* — the loss of position-dependence, and
exactly what distilling a teacher that reads 0.0000 at every probe position produces.

Both arms, monotone in lr:

| arm | lr | code flips | KLD (from 0.0762) | recovered | control |
|---|---|---|---|---|---|
| 1 | 2e-4 | ~2% | 0.2724 | **-257.7%** | 0.0043 |
| 2 | 5e-4 | 5.6-6.2% | 0.3576 | **-369.5%** | 0.0000 |

More lr -> more code flips -> MORE damage and worse termination. The flips are real
learning (arm 2 even shows the first true sign reversals, `±->∓: 8`, where arm 1 had
none) — they are just learning the wrong target. This is what "training is walking
toward a different model" looks like from the dense model's point of view.

Arm 3 (1e-3) was cancelled: it would trace the same curve faster.

### What four passing checks failed to catch

The 31B teacher passed every gate this pipeline has, and was still unusable:

* tokenizer identity — **262,144/262,144** ids, every control token included
* corpus fingerprint, all 651 windows present, forced stop id `[106]`
* support coverage **0.9993**
* a **4,090x** discriminative stop-signal ratio at the corpus's real stop targets

All four are properties of the TABLE. None of them asks whether the teacher's policy is
the one the student should adopt, and the one instrument that does — the teacher's own
stop probe, read through the student's template — was reading 0.0000 at every point
before a single training step ran. It was in the log; it was not a gate. **It should be
one.**

## CE-only inverts the story, and exposes a flaw in MY metric

| arm | teacher | lr | KLD | recovered | control @50 | flips |
|---|---|---|---|---|---|---|
| untrained ternary | — | — | 0.0762 | — | 0.0734 | — |
| arm 1 | 31B KD | 2e-4 | 0.2724 | -257.7% | 0.0043 | ~2% |
| arm 2 | 31B KD | 5e-4 | 0.3576 | -369.5% | 0.0000 | 5.6-6.2% |
| **CE-only** | **none** | 2e-4 | **0.3866** | **-407.7%** | 0.0022 | 2.4-2.7% |

**CE-only is the worst arm.** Removing the teacher made BOTH damage and termination
worse, so the foreign teacher was restraining the drift rather than causing it. The
earlier conclusion — "the 31B teacher caused the collapse" — is wrong. It accelerates it
(control 0.0041 by step 25 vs CE's 0.0453) and then holds a floor; CE-only decays past it.

### The metric cannot answer the question it was written for

`KLD(dense ‖ candidate)` on held-out text sums two things that a fine-tune does at once:

1. ternarization damage failing to recover, and
2. the model legitimately learning the training distribution.

Every fine-tune does (2) whether or not it does (1), so **every "recovered -X%" figure
above is contaminated**, and the ordering across arms measures how hard each one trained
rather than how well it recovered. The 5x gap between untrained (0.0762) and CE-only
(0.3866) is mostly (2).

This is a flaw in the experimental design, not a surprise in the data. `notes.md`
pre-registered the self-KD arm for the TEACHER confound and did not pre-register a
control for this one.

### The missing control, now queued

Train the SAME six layers with **no ternarization** — identical corpus, lr, steps —
expressed as `--dense-kind _proj --dense-kind gate`, which leaves every linear trainable
but off the grid (`wrap_model` refuses a trainable layer that is not also ternarized, so
this is how a dense arm is written). Then:

* if the dense arm also lands near 0.38, ternarization is not what these numbers
  measured, and the honest reading of tonight is that the instrument was wrong;
* if it lands near 0.08, training really does amplify ternarization damage and the
  original NO-GO reading stands.

`gemma4_stage_damage.py --ref-ckpt` then measures `KLD(dense fine-tune ‖ ternary fine-tune)`
directly — the same layers, the same data, the same number of steps, differing only in
whether the weights were on the grid. That is the quantity the study meant by "damage"
all along.

Its abort guards are deliberately disabled (`--probe-abort 0`): a dense arm must be
allowed to show a termination collapse, because if a DENSE fine-tune on this corpus also
drives the control to ~0.004, then termination is a property of the corpus and the
schedule, not of ternarization — which would be the single most useful thing learned
tonight.
