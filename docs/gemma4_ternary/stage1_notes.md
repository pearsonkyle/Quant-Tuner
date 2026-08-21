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

Three 60-step arms (`EPOCHS=0.37` at `GRAD_ACCUM=4` over 651 windows), identical but for
lr: **2e-4 / 5e-4 / 1e-3**. Each is read on four things, and no single one decides:

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
declined to ternarize. This does not change stage 1's question -- which is whether QAT
recovers damage -- but it changes what shipping looks like: `down_proj` wants to be
4-bit, not dense, and ideally quantization-aware at 4 bits during training rather than
left in bf16 and quantized after. The export mechanism already exists
(`llama-quantize --tensor-type`, same lever as `quantize.mtp_pin`).

Flagging now so the number is on the record before the schedule is committed to.
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
