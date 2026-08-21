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
