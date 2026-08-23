# gemma-4-E4B ternarization — handoff (paused 2026-08-22 16:30 UTC)

GPU released mid-study. This is everything needed to resume on another machine.

## Where the study actually stands

**The stage-1 capability question is answered, and the answer is yes.** At matched
training — same layers, corpus, lr, steps and seed, no teacher in either arm, differing
only in the grid — held-out masked CE was **dense 1.7796 vs ternary 1.7290**. Ternarizing
stage 1 (layers 0,1,2,3,7,8 with `down_proj` held dense) costs nothing measurable.

**What is not answered is termination**, and the reason it stayed open is that the
metric the brief pre-registered turned out to be invalid, twice over:

1. *KLD vs the shipped model measures the wrong thing.* A **dense** fine-tune on this
   corpus moves 0.2175 from shipped all by itself, so KLD-vs-shipped scores fine-tuning
   drift and ternarization damage as one number, dominated by the drift. Every
   "recovered −258% / −370% / −408%" figure from the early arms is that artifact. The
   The apparently valid form — KLD against a **dense fine-tune** trained identically —
   was measured after the pause and **also fails**, for a deeper reason. See "Both
   in-flight jobs landed" below; the short version is that KLD from a dense reference
   measures divergence rather than damage, and on stage 1 the ternary arm is the better
   model of the two.
2. *The 7-prompt synthetic stop probe is unreliable in both directions here.* It called
   untrained-ternary unchanged and a dense fine-tune improved. Replaced by
   `scripts/gemma4_stop_on_corpus.py`, which samples real stop targets out of the
   held-out corpus, gives each 2,048 tokens of its own context, and reads P(stop) at the
   position that predicts it. **This is the instrument to trust.**

### The measurement (`scripts/gemma4_stop_table.py`)

| arm | commit | P(stop)@stop | elsewhere | ratio | val CE |
|---|---|---|---|---|---|
| shipped | 35.0% | 0.3452 | 1.7e-04 | 2054 | — |
| untrained-ternary | 15.0% | 0.2418 | 6.4e-04 | 379 | — |
| dense-ft | 2.5% | 0.1327 | 1.1e-03 | 123 | — |
| ce-only | 0.0% | 0.0265 | 2.9e-04 | 91 | — |
| self-KD (α 0.5) | 2.5% | 0.0788 | 9.1e-04 | 87 | 1.9948 |
| self-KD + stop-weight 5.5 | 5.0% | 0.0781 | 4.3e-05 | 1798 | 2.0478 |
| self-KD + stop-weight 16 | 17.5% | 0.2706 | 6.4e-04 | 419 | 2.6457 |

Read it as **two independent failures**, not one:
- **commitment** (`frac_top1`) — is stopping the *top* choice at a real stop target.
- **discrimination** (`ratio`) — P(stop) there over P(stop) at ordinary positions.

They move independently. stop-weight 5.5 took the ratio 87 → 1,798 and left commitment
at 5%. Reporting them blended would have scored that a win.

### Three things that are settled, so don't re-litigate them

- **The commitment deficit is not quantization-specific.** A dense fine-tune lands at the
  same 2.5% as ternary self-KD. Whatever destroys commitment does it to a bf16 model too.
- **stop-weight only moves the model along a capability/commitment curve.** This is the
  central result and it superseded an earlier, wrong summary of mine ("16 bought
  neither"): stop-weight 16 read the *highest* commitment of any trained arm, 17.5%. But
  all four of its stop metrics land on **untrained-ternary** (15.0% / 0.2418 / 6.38e-04 /
  379 — at n=40 that is 7 targets vs 6, i.e. noise), because what it actually did was
  decline to learn: unweighted val CE 2.7072 → 2.6457 against 5.5's 2.4918 → 2.0478, and
  20–25% fewer code flips.

  | arm | final val CE | flips (L0 q_proj) | commit |
  |---|---|---|---|
  | untrained-ternary | — | 0% | 15.0% |
  | stop-weight 16 | 2.6457 | 2.28% | 17.5% |
  | stop-weight 5.5 | 2.0478 | 2.72% | 5.0% |
  | self-KD (100 steps) | 1.9948 | 4.14% | 2.5% |
  | dense fine-tune | 1.7796 | (dense) | 2.5% |
  | CE only | 1.7290 | 2.65% | 0.0% |

  **The ordering by val CE is the exact reverse of the ordering by commitment**, and it
  does not care about the grid — the dense fine-tune sits next to CE-only. Fitting this
  corpus destroys the stopping policy in proportion to how well you fit it. Validation is
  unweighted masked CE (`run_validation` passes no `weights=`), so these are comparable.
- **The corpus is not at fault, and I checked rather than assumed.** 13,273 `<end_of_turn>`
  tokens are present, **6,300 labeled and 6,973 correctly masked** (the ends of user/tool
  turns). gemma stops at every assistant turn including tool calls and the corpus labels
  exactly those. 1 stop per 972 supervised tokens is what this data's assistant turns
  genuinely average — so re-balancing it by weight would teach stopping *more often than
  the dialect calls for*. That is the argument for a saturating objective over a weighted
  one.
- **The 31B teacher is refused, permanently.** It passed every table-level check
  (tokenizer 262,144/262,144, coverage 0.9993, stop ratio 4,090×) and was still unusable:
  its own stop probe reads 0.00003 where the student reads 0.07032. `run_gemma4_selfkd_arm.sh`
  now gates on this. Training distils against the **shipped E4B itself**.

## The next experiment, already written and never run

`scripts/run_gemma4_anchor.sh` — **this is the thing to run first on the new machine.**

Every KD arm has carried `--stop-anchor 0.2`, i.e. inert. Its own telemetry says the
mechanism works and the gain is ~100× short: `an` climbs 0.0008 → 0.15–0.47 across 60
steps (the student drifting off the teacher's stop level, the hinge engaging) while
contributing 0.2 × 0.2 = 0.04 against a loss of 1.2–2.0.

The anchor is the right *shape* where stop-weight is not. It is **one-sided and
saturating**: it pushes only until the student is within `margin_hi` (0.1 nat) of the
*teacher's own* log P(stop) at that position, then goes silent. The teacher is the shipped
model, whose stopping policy is precisely what we are trying not to destroy
(corpus-conditioned: **0.530** at the 6,300 real stop targets, 0.00053 at the other
6,118,196 rows, ratio 993 — `kd/stop_signal_e4bself.json`). So a large β converges to
shipped behaviour instead of overshooting into a stop-happy model.

Two arms, 60 steps each, ~50 min apiece: β=8 and β=25, both with stop-weight 5.5.
`an` is reported raw (pre-β) in the step line, and loss adds `β · an`.

**Pre-registered criteria** (sharpened after sw16 — a bare commitment bar is clearable by
under-training, so val CE is now part of the pass condition, not a side note):
- **Pass** — commitment ≥ 15% **and** val masked-CE ≤ 2.10 and ratio ≥ 1000. That is
  *off* the curve above, not further along it.
- **Partial** — commitment 8–15% at val ≤ 2.10. Carry into the full stage with larger β.
- **Fail** — commitment < 8%, or val > 2.15, or commitment bought at a val CE that puts
  the arm back on the curve.

A saturating anchor is the one thing tried so far that *could* leave the curve: it adds
force only at stop targets and only until the student reaches the teacher's own level
there, so unlike a 16× CE reweighting it has no mechanism for degrading general fit. If
it too only walks the curve, the honest conclusion is that this corpus and this objective
cannot both be satisfied, and the next move is a mixed objective — holding the shipped
model's full distribution at stop targets while fitting content elsewhere — not another
scalar.

If an arm passes: `scripts/run_gemma4_stage1_full.sh` runs stage 1 at full length
(1 epoch = 651 steps, ~8.5 h) with a CPU sidecar that reads commitment off each
checkpoint while the GPU stays full — masked-CE cannot see this failure (sft32k's
validation went flat for 225 steps while its stopping policy collapsed), so commitment
falling across checkpoints while val improves is the signal to kill the run.

```bash
RECIPE_ARGS="--stop-anchor 8 --stop-weight 5.5" TAG=anchor8 \
  bash scripts/run_gemma4_stage1_full.sh
```

## What to move to the new machine

Everything under `out/gemma4-ternary/` is **60 GB** and most of it is reproducible. The
irreducible set is **2.7 GB**:

| path | size | rebuild cost if dropped |
|---|---|---|
| `corpus_sft_gemma4_32768.pt` | 326 MB | ~20 min CPU |
| `corpus_sft_gemma4_val_32768.pt` | 44 MB | ~5 min CPU |
| `kd/e4bself_topk64_fs106.pt` | 2.3 GB | **several GPU-hours** — move this |
| `layer_damage.json`, `stop_baseline.json`, `kd/stop_signal_*.json`, `stopcorpus/*.json` | < 1 MB | hours of CPU |

The corpus fingerprint is `0c70d992882d29a7` and the KD table is bound to it — a table
from a different corpus is refused at startup, which is the check you want, so move the
corpus and the table together or rebuild both.

**Checkpoints are 2.1 GB each and nine of them exist (19 GB).** Only two are worth
carrying: `dense-control-lr2e-4/trained_latents.pt` (the reference for every valid damage
measurement) and `sw5.5-lr2e-4/trained_latents.pt` (best termination so far). The rest are
superseded diagnostics.

Not on this machine and re-downloadable: `google/gemma-4-E4B-it-qat-q4_0-unquantized`.
`LLAMA_CPP_DIR=vendor/llama.cpp-prism` is needed only for Q2_0 export (ftype 41 is
fork-only), which this study has not reached.

## Day 1 on the new machine

Branch is **`qat/ternary-32k-cuda`**. Note that another session has been committing to it
in parallel (exp-059, a separate Qwen-coder QAT study) — rebase rather than assume the
tip is where this study left it, and nothing below shares state with exp-059 except the
`src/quant_tuner/qat/` code.

**Hardware.** The 60-step arms ran at ~70 GB of GPU memory (the trainer's own report is
`mem=30.3/62.0GiB` for tensors; nvidia-smi showed 69.9 GB resident). Budget an 80 GB card.
fp32 latents and fp32 compute are not optional here — bf16 underflows the ternary
threshold and no codes flip.

```bash
git checkout qat/ternary-32k-cuda && git pull --rebase
uv sync
hf download google/gemma-4-E4B-it-qat-q4_0-unquantized   # ~16 GB, forward-only teacher too

# copy from the old box — `out/` is GITIGNORED, so none of this travels with the repo.
# Bundle exactly the irreducible set (2.7 GB, verified) on the OLD machine:
#
#   tar -czf /tmp/gemma4-ternary-resume.tgz \
#       out/gemma4-ternary/corpus_sft_gemma4_32768.pt \
#       out/gemma4-ternary/corpus_sft_gemma4_val_32768.pt \
#       out/gemma4-ternary/kd/e4bself_topk64_fs106.pt \
#       out/gemma4-ternary/kd/stop_signal_e4bself.json \
#       out/gemma4-ternary/layer_damage.json \
#       out/gemma4-ternary/stop_baseline.json \
#       out/gemma4-ternary/stopcorpus out/gemma4-ternary/stage1
#
# then untar at the repo root on the new one. Contents:
#   out/gemma4-ternary/corpus_sft_gemma4_32768.pt
#   out/gemma4-ternary/corpus_sft_gemma4_val_32768.pt
#   out/gemma4-ternary/kd/e4bself_topk64_fs106.pt
#   out/gemma4-ternary/{layer_damage,stop_baseline}.json
#   out/gemma4-ternary/kd/stop_signal_e4bself.json
#   out/gemma4-ternary/stopcorpus/            (all the read-outs; < 1 MB, hours of CPU)
#   out/gemma4-ternary/dense-control-lr2e-4/trained_latents.pt   (2.1 GB, optional)
```

**Verify the transfer before spending a GPU-hour on it.** The KD table is bound to the
corpus by fingerprint `0c70d992882d29a7`; a table from a different corpus resolves fine
and distils against the wrong distribution at every position.

```bash
uv run python scripts/verify_kd_table.py \
    --table out/gemma4-ternary/kd/e4bself_topk64_fs106.pt \
    --corpus out/gemma4-ternary/corpus_sft_gemma4_32768.pt
# expected — verified on the old box 2026-08-23:
#   corpus fingerprint  0c70d992882d29a7  (651 windows, all present: True)
#   forced ids          [106]
#   support coverage    0.9917
#   topk=64, positions=6,124,496, windows=651
#   [kd-verify] OK
uv run pytest tests/unit/test_qat_wrap_model.py -q     # 12 tests, no model files needed
uv run python scripts/gemma4_stop_table.py             # should reprint the table above
```

**Then run the experiment that is already written and was never started:**

```bash
bash scripts/run_gemma4_anchor.sh      # two 60-step arms, beta 8 and 25, ~50 min each
uv run python scripts/gemma4_stop_table.py
```

Judge it on the sharpened bar — commitment ≥ 15% **and** val masked-CE ≤ 2.10 — not on
commitment alone, because sw16 already cleared a bare commitment bar by under-training.

If an arm passes, the full stage is one command and about 8.5 hours:

```bash
RECIPE_ARGS="--stop-anchor 8 --stop-weight 5.5" TAG=anchor8 \
    bash scripts/run_gemma4_stage1_full.sh
```

Keep `bash scripts/gemma4_report_watch.sh 300` running alongside it (the argument is
the refresh interval in seconds, not a path) so
`report.html` stays live, and **queue the next arm before the current one ends** — an
unqueued gap cost 6.8 h of idle card on 2026-08-22.

## Both in-flight jobs landed — and the damage metric is the third one to fail

The sw16 stop read-out is folded into the table above; it is what produced the
capability/commitment curve.

**The damage measurement completed and does not answer the brief's question.** Against
the dense fine-tune reference (12 windows, split `test`):

| row | KLD vs dense-ft | top-1 | ppl |
|---|---|---|---|
| shipped | 0.3173 | — | 4.566 |
| ternarized shipped ("untrained") | 0.2635 | 0.870 | 4.359 |
| trained ternary (self-KD, 100 steps) | 0.3927 | 0.826 | 4.284 |

The harness printed "recovered −49.1%", and **that number is not interpretable** — the
two rows do not sit on the same footing. `trained` is (trained ternary ‖ trained dense):
both saw the same data, so the fine-tuning drift largely cancels and what is left is
close to ternarization alone. `untrained` is (ternarized *shipped* ‖ trained dense),
which still carries the entire drift, because the reference moved and that candidate did
not. Their ratio is not a recovery fraction. `gemma4_stage_damage.py` no longer emits one
under `--ref-ckpt`, and now also reports the reference's own ppl so the file can answer
the question its KLD column invites.

The deeper problem is that **KLD-from-a-dense-reference measures divergence, not damage**,
and on stage 1 the ternary arm is not the worse model: ternarizing lowers ppl on this
probe (4.566 → 4.359, consistent across two independent window sets), the trained ternary
is the best of the three reported (4.284), and the matched held-out CE comparison has the
ternary arm ahead (1.7290 vs 1.7796). A large KLD between two models of equal quality is
difference, not injury.

So all three metrics the brief pre-registered have now failed for the same underlying
reason: **each assumed the shipped or dense model is the ceiling, and on stage 1 it is
not.** The measurement that survives is capability against a matched control (held-out
masked CE), plus the real-corpus stop read-out for the policy the CE cannot see.

**`a75-sw5.5-lr2e-4` was stopped at step 21 of 60** to free the GPU. The trainer caught
the signal and saved cleanly (`trained_latents.step21.pt`, with flip telemetry), but 21
steps is not comparable to the 60-step arms — treat the arm as **not run**. Its question
(does kd-alpha 0.75 pull commitment toward the dense distribution?) is superseded by the
anchor arms and is not worth re-running first.

## Traps that have already cost time here

- **Never `pgrep`/`grep` for a pattern your own command line contains** — the wait loop
  matches itself. Cost 1.6 h of idle GPU once. Use exact pids, or a condition on GPU
  memory. (It bit again during this very shutdown.)
- **Never edit a `.sh` a live bash is executing** — byte-offset corruption. Check with
  `ps -eo args` first.
- **Queue the next arm before the current one ends.** An unqueued gap cost 6.8 h of idle
  GPU on 2026-08-22.
- `--stop-weight` needed `resolve_vocab_size` — `Gemma4Config` has no flat `vocab_size`.
- gemma's vision/audio towers contain submodules literally named `linear`, so any
  name-based latent selection finds 280 where wrapping produced 48. Select by module
  **type**; pinned by `test_latent_modules_counts_wrapped_latents_not_names`.
- flash-attn is structurally unavailable for this model: FA2 caps head_dim at 256 and
  gemma-4 uses `global_head_dim 512`. `attention.enable_gqa_repeat_where_unfused()`
  keys off head_dim as well as dtype for exactly this reason.

## Live report

`scripts/gemma4_report.py --out out/gemma4-ternary/report.html` regenerates it from the
JSONs above; `gemma4_report_watch.sh` keeps it live during a run. Published at
https://claude.ai/code/artifact/3a232be9-e4ac-469c-b44c-ddb48281782e
