# 32K ternary-QAT: state of play and how to resume elsewhere

Written 2026-08-14 after the work below crashed the M4 Max it was running on (see
"The crash" — it was an operator error, not a property of the code). Everything here is
committed; nothing is left in a working tree.

Target: `prism-ml/Ternary-Bonsai-8B` continued QAT, all-36 layers, fp32 latents, Adafactor.
Reference box: **M4 Max, 128 GB unified memory, macOS/MPS, torch 2.12, transformers 5.8.1.**
Every number below is from that box; **none of it transfers to CUDA unchanged** (see
"Porting to a different machine").

> **2026-08-16: ported and re-measured on CUDA.** See §10 — it supersedes §3's ladder and
> corrects §9's central prediction. The short version: CUDA does **not** hand fp32 a fused
> attention kernel, chunked SDPA is pure overhead here, 32768 full-gradient fits with
> 6.4 GiB to spare, and one epoch of the universal corpus costs ~10 h instead of ~152 h.

---

## 1. Why this work happened

The previous run (`sft8k-full`, 8064 window, 522 steps) trained cleanly and moved
*behaviour* but not *capability*: 0/10 resolved before and after, tool-error rate 0.65 →
0.33, but it hit `max_turns` on 7/10 instances and looped on 97% of trajectories.

Two hypotheses, both now addressed in code:

1. **The window was too short to teach termination.** At 8064 only 27% of SWE
   trajectories fit whole, so the model rarely saw a complete task→completion arc.
2. **The stop token is drowned.** 5,740,167 trainable tokens against 32,448 terminating
   `<|im_end|>` targets — one "stop" decision per 176 "keep going".

---

## 2. The finding that mattered

**Query-chunked SDPA capped one attention block's score tensor but not their SUM.**

With grad enabled every block saves its own softmax output for backward, so the saved
total is the whole `[heads, q_len, kv_len]` matrix regardless of block size:

| window | attention weights saved for backward |
|---|---|
| 8064 | 7.8 GiB — why this never surfaced |
| 16128 | **31 GiB** |
| 32768 | **128 GiB** |

That, and nothing else, is why the docs said "16128 completes no step in 31 minutes at 99%
swap". It was never a fundamental limit.

`chunked_causal_sdpa(recompute_scores=True)` (default on, inert without grad) checkpoints
each block so the scores are recomputed in backward. Forward is bit-identical, gradients
match to 1e-6, and a unit test counts SDPA calls across forward/backward — if the
recompute silently stops firing, forward and gradients stay *correct* and the only symptom
is that long windows OOM again.

---

## 3. The measured ladder

Real training loop (optimizer step included, not a single-window probe), all-36 / fp32 /
adafactor, grad-accum 2, idle box after a 150 s cooldown, scores recomputed:

| config | s/step | ms / **trained** token | vs 8064 | **targets trained** | SWE convs whole | swap Δ |
|---|---|---|---|---|---|---|
| 8064 full-gradient | 178 | 11.1 | 1.0x | 100% | 27% | flat |
| 16128 full-gradient | 626 | 19.4 | 1.8x | 100% | 68% | +2.6 GB |
| 20480 full-gradient | 1061 | 25.9 | 2.3x | 100% | 81% | +1.5 GB |
| 24576 full-gradient | 1372 | 27.9 | 2.5x | 100% | 88% | +2.1 GB |
| **32768 full-gradient** | **1792** | **27.3** | **2.5x** | **100%** | **97%** | **−3.9 GB** |
| 32768, tail 16384 | 1490 | 45.5 | 4.1x | 54% | 97% | −0.2 GB |
| 32768, tail 8192 | 1790 | 109 | 9.9x | 26% | 97% | +1.0 GB |
| 32768, tail 24576 | — | — | — | — | — | OOM 133.8 GiB |

Raw JSON in `out/probe/r*.json`; regenerate with `scripts/probe_window_budget.py`.

**Conclusions:**

- **Train the full window. Do not use a prefix at 32768.** Cost per trained token is flat
  from 20480 to 32768 (25.9 / 27.9 / 27.3 ms) — quadratic attention is still small next to
  the linear layers at 8B, so extra context is nearly free. Take the longest that fits.
- **A prefix discards training signal, not just compute.** Targets are spread evenly
  through a window, so a tail of T out of W trains ~T/W of them. Measured on the 32768 SWE
  corpus (26 windows, 267,392 labeled targets): tail 8192 → 25.9%, tail 16384 → 53.7%.
  `--trained-tail` is validated **bit-exact** but is now a fallback for going *past* the
  full-gradient activation ceiling, not the plan.
- **bf16 is unusable at long windows on Metal**: par with fp32 at 8064 (161 vs 151
  ms/layer), then **68,346 ms vs 2,456 ms at 32768** (0.1 vs 3.6 TFLOP/s). A pathological
  MPS path, not a gradual falloff.
- **There is no fused kernel to borrow.** torch's `is_causal` is not fused on MPS — at
  32768 it asks for a 128 GiB buffer. Our chunked path is faster than it at every size.
- **An 8-bit optimizer is a no-op here.** Adafactor's factored state is 65 KB for a 201 MB
  tensor (**0.03%**), ~9 MB across all 6.95 B trainable params. Memory is params (32.8 GB)
  + grads (27.8 GB) + activations.

---

## 4. What is validated

- `scripts/validate_prefix_context.py` on the real 36-layer model: prefix-split loss vs a
  full-window forward over the same target set — **delta 0.00e+00 on 4/4 windows**, i.e.
  bit-exact. Run this as a gate before any long job.
- 915 unit tests pass. New coverage: offset-aware SDPA (cached prefix), score recompute
  (forward/gradient parity + a call-counting test), prefix-context under gradient
  checkpointing, stop-token weighting and its `sum(w[target])` denominator.
- **Live 32768 training step on the real SWE corpus**: `loss=0.6019 gnorm=1.29
  mem=31.5GiB 1799.6s/step` — matching the probe's 1791.6 exactly. Artifacts in
  `out/exp-058/smoke32k/{metrics.jsonl,watch.log}`.

---

## 5. The crash

`/tmp/valcost.py` (a throwaway script, not in the repo) loaded the 30 GB model **~3
seconds after `pkill`-ing the training process**, while the system still held that
process's pages and ~29 GB of swap. The box went down during model load; the log stops
after `Loading weights: 100%` with no timing printed.

**This was operator error, not a code or configuration finding.** `scripts/probe_window_budget.py`
already has `--cooldown` (default 120 s) precisely for this — a config launched while
macOS still holds the previous one's memory gets killed during load and reads as a false
OOM. I did not apply the same discipline to a hand-rolled script.

**Rule for the next box: wait for `sysctl -n vm.swapusage` to settle before loading the
model.** Nothing about the 32768 training configuration is implicated — it ran for 110
minutes at a flat 31.5 GiB with swap *declining*.

---

## 6. The open question this was trying to answer

**How expensive is validation at a 32768 window?** The slow monitor caught that the val
interval was the only place swap moved (+11.2 GiB in one 5-minute sample) and that it
dragged the surrounding steps with it. Step 1 took 30 minutes; step 2 plus a 3-window
validation had not finished 80 minutes later.

Already fixed in `15c1884`: release the MPS cache before *and* after validation, time it,
and print `(N windows in Xs)` plus a `val_seconds` field in `metrics.jsonl`. So the next
run **measures this itself** — the standalone script is no longer needed.

Unresolved: the right `--val-windows` at 32768. The default of 16 is very likely too
expensive. Start at **4** and read the printed cost against the step time.

---

## 7. Resume here

```bash
# Corpora already built and on disk (rebuild with the commands below if starting fresh):
#   out/exp-058/sft_corpus_swe_32768.pt      26 windows, 878,010 tok, 1,915 im_end targets
#   out/exp-058/sft_corpus_val_32768.pt      81 windows, 2,774,117 tok

PYTHONPATH=src python scripts/build_sft_qat_corpus.py \
    --sft out/corpora/qwen3-universal/sft.jsonl.gz --source swe-trajectories \
    --window 32768 --max-tool-tokens 8192 --min-density 0.05 \
    --out out/exp-058/sft_corpus_swe_32768.pt
PYTHONPATH=src python scripts/build_sft_qat_corpus.py \
    --sft out/corpora/qwen3-universal/sft.jsonl.gz --split test \
    --window 32768 --max-tool-tokens 8192 --min-density 0.05 \
    --out out/exp-058/sft_corpus_val_32768.pt

# 1. GATE — do not skip; it is ~5 minutes against a 26 h run
PYTHONPATH=src python scripts/validate_prefix_context.py \
    --corpus out/exp-058/sft_corpus_universal_8064.pt --tail 4096 --windows 4

# 2. Re-measure one step on the new hardware before committing to the full run
PYTHONPATH=src python scripts/probe_window_budget.py --window 32768 --trained-tail 0 \
    --steps 2 --grad-accum 2 --cooldown 150

# 3. The run
PYTHONPATH=src python -m quant_tuner.qat.train \
    --corpus out/exp-058/sft_corpus_swe_32768.pt \
    --val-corpus out/exp-058/sft_corpus_val_32768.pt \
    --train-layers 36 --optim adafactor --dtype fp32 \
    --grad-accum 1 --epochs 4.0 --lr 5e-4 --warmup-frac 0.05 \
    --stop-weight 6.0 --val-every 10 --val-windows 4 \
    --ckpt-every 15 --ckpt-keep 3 \
    --out out/exp-058/swe32k

# 4. Monitor on a slow interval — do NOT poll the process
bash scripts/watch_qat_run.sh out/exp-058/swe32k 300 48
```

`--grad-accum 1`, not 2: 26 windows at accum 2 leaves only 13 steps/epoch — too few for
the cosine schedule or the grad-spike guard's 20-step median window.

Expect ~900 s/step at accum 1, 26 steps/epoch, **~26 h for 4 epochs** on the reference box.
Healthy steady state is `mem=31.5GiB` with swap flat or declining. Read `gnorm` and the
flip telemetry, **not the loss** — a ternary model only learns by flipping codes, and the
loss falls on scale drift alone.

---

## 8. Decisions still open (deliberately not made)

1. **Corpus scope.** SWE-only is ~26 h for 4 epochs but risks eroding general behaviour —
   the last run's one real gain was tool-error rate. Mixing in `logs-agents` costs
   proportionally more wall-clock. Not my call to assume.
2. **`--val-windows`.** Suggest 4; confirm from the cost the run now prints.
3. **The holdout cannot settle the result.** At n=10 with 0/10 both, the 95% upper bound
   on the true resolve rate is ~31%. Enlarge the SWE-rebench holdout before the A/B, or a
   real improvement will not be distinguishable from noise.
4. **`swe-trajectories` has no rows in the `test` split**, so the val corpus is
   agentic-log distribution, not held-out SWE. The real gate stays the SWE-rebench holdout.

---

## 9. Porting to a different machine

The *code* changes are hardware-agnostic and are wins anywhere:
`recompute_scores` cuts real memory on CUDA too, and prefix-context is exact everywhere.

The *numbers* are not. Re-derive them:

- The MPSGraph `INT_MAX` element cap and `DEFAULT_SCORE_BYTES` budget are Metal-shaped.
  On CUDA, PyTorch has a genuinely fused SDPA (FlashAttention) that never materializes the
  score matrix, which makes `chunked_causal_sdpa` **unnecessary and slower** — check
  whether `enable_chunked_sdpa()` should be off entirely (`--no-chunked-attention`).
  With flash attention available, the whole memory problem this document solves largely
  evaporates and much longer windows should be reachable.
- `foreach=False` and the bf16 findings are MPS quirks. On CUDA, bf16 is the normal choice
  and `--compute-dtype bf16` with fp32 masters should be re-tested, not assumed bad.
- Adafactor was chosen because AdamW's 55.6 GB of state did not fit in 128 GB shared.
  On an 80 GB+ dedicated GPU, revisit AdamW.
- Re-run step 2 of the resume block to get s/step before sizing the run.

---

## 10. The CUDA port (2026-08-16)

Box: **1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 95.0 GiB VRAM, cc 12.0**,
driver 590.48.01, torch 2.12.0+cu130, transformers 5.12.1, python 3.11.

### 10.1 The finding that mattered — fp32 has no fused attention kernel on CUDA

§9 predicted that "PyTorch has a genuinely fused SDPA (FlashAttention) … which makes
`chunked_causal_sdpa` unnecessary and slower" and that "the whole memory problem this
document solves largely evaporates". **Half right, and the wrong half is load-bearing.**

The first fp32 run on this box OOM'd at a **8064** window on a 95 GiB card, asking for
7.75 GiB. That number is exactly `32 heads x 8064^2 x 4 bytes` — the full score matrix,
i.e. the very tensor chunking exists to avoid, materialized on the hardware that was
supposed to make chunking unnecessary.

The cause is a dispatch fallback, not a missing kernel:

- FlashAttention and cuDNN attention **reject fp32 outright** (`Expected query, key and
  value to all be of dtype: {Half, BFloat16}`).
- The **memory-efficient** kernel *does* support fp32 — but not together with
  `enable_gqa`.
- transformers passes `enable_gqa=True` to SDPA whenever the attention mask is None on
  CUDA (`integrations/sdpa_attention.py::use_gqa_in_sdpa`) instead of expanding K/V
  itself. Qwen3-8B is 32 heads / 8 KV, so that branch is always taken.

With no kernel able to serve fp32 + GQA, the dispatcher falls all the way back to
**math**, which materializes `[batch, heads, S, S]` for backward. Measured at S=2048,
fp32, one fwd+bwd (`[H,S,S]` would be 0.500 GiB):

| call shape | peak | kernel |
|---|---|---|
| `repeat_kv` + `is_causal` | 0.251 GiB | fused |
| `enable_gqa=True` + `is_causal` | 2.156 GiB | **math** |
| `repeat_kv` + explicit bool mask | 0.266 GiB | fused |

`attention.enable_fp32_gqa_repeat()` makes the predicate honest about fp32 so
transformers' own `repeat_kv` branch is taken; `train.py` calls it on CUDA whenever
`compute_dtype` is fp32. That is what turns "8064 OOMs" into "32768 fits".

**Consequence for §9:** `--no-chunked-attention` alone was never the answer on CUDA. The
chunked path was masking a dispatch bug, not merely working around an MPSGraph cap.

### 10.2 The measured ladder — this replaces §3

Same protocol as §3: `scripts/probe_window_budget.py`, the real training loop with the
optimizer step, all-36 / adafactor / grad-accum 2 / 3 steps, synthetic corpus at
`--labeled-frac 0.35`. "fused" = stock SDPA with the fp32 GQA fix; "chunked" =
`--chunked-attention on`. `peak` is `torch.cuda.max_memory_allocated`.

| window | fp32 fused | fp32 + TF32 | fp32 chunked | bf16 fused | bf16 chunked |
|---|---|---|---|---|---|
| 8064  | **20.1 s** / 64.8 GiB | **11.8 s** / 64.8 GiB | 27.3 s / 69.3 GiB | **5.5 s** / 67.8 GiB | 17.4 s / 67.8 GiB |
| 16128 | **45.2 s** / 72.6 GiB | — | — | **10.8 s** / 67.9 GiB | — |
| 32768 | **124.2 s** / 88.6 GiB | **90.1 s** / 88.6 GiB | 228.6 s / 88.6 GiB | **24.7 s** / 70.9 GiB | — |
| 65536 | **OOM** | — | **66.5 s** / 87.4 GiB | — |

**The ceiling is 65536, and it is bf16-only.** 65536 is the model's
`max_position_embeddings`, so it is also the last rung that exists. fp32 cannot reach it
(fp32 is already at 88.6 GiB of 95.0 at 32768); bf16 gets there with **7.6 GiB spare —
and that figure excludes validation and the checkpoint transient**, so it is not a
configuration to run without a real pre-flight. Note also that bf16's cost per trained
token stops being flat there: 0.32 / 0.32 / 0.37 / **0.51 ms** at 8064 / 16128 / 32768 /
65536. Doubling the window past 32768 costs 38% more per token and buys SWE
conversations-whole from 97% to 100%.

Before the fp32 GQA fix, `fp32 fused` **OOM'd at every window including 8064**.

Cost per trained token (per-window fwd+bwd, netting out the ~0.4-0.5 s fixed per-step
cost — note that fixed cost was **~106 s** on the M4 Max and is essentially free here):

| window | fp32 | bf16 | MPS fp32 (§3) |
|---|---|---|---|
| 8064  | 1.22 ms | 0.32 ms | 11.1 ms |
| 16128 | 1.39 ms | 0.32 ms | 19.4 ms |
| 32768 | 1.89 ms | 0.37 ms | 27.3 ms |

**Conclusions, and where they differ from §3:**

- **Train the full window at 32768. Still no prefix.** §3's headline conclusion survives:
  extra context remains cheap relative to the linear layers. But the CUDA slope is not
  flat the way MPS's 20480-32768 plateau was — fp32 costs **55% more per token** at 32768
  than at 8064 (bf16, on flash, is nearly flat at +16%). It is still overwhelmingly worth
  it, because a prefix discards ~T/W of the targets and this buys 27% -> 97% of SWE
  conversations whole.
- **Chunked SDPA is a Metal workaround. It is pure overhead here** — 3.2x on bf16 at 8064
  (5.5 -> 17.4 s), and 26% slower plus 4.5 GiB heavier than the fp32 fused path. It is now
  opt-in per backend (`--chunked-attention auto|on|off`), enabled automatically only on
  MPS and for `--trained-tail` on any device.
- **bf16 compute is a 5x speedup and 17.7 GiB lighter at 32768** — the exact opposite of
  the Metal finding in §3, where bf16 was a pathological 28x pessimization. Its memory is
  also nearly flat in the window (67.8 -> 70.9 GiB from 8064 to 32768) because the
  `MasterOptimizer` static footprint dominates and bf16 activations are half-size, whereas
  fp32's lower static footprint (56.4 GiB) is swamped by full-size activations.
- **`--optim adamw` still does not fit, now for a VRAM reason rather than a shared-memory
  one.** Static footprint before any activations, against 95.0 GiB:

  | configuration | params | grads/masters | optimizer state | total | |
  |---|---|---|---|---|---|
  | fp32 + adafactor | 30.5 | 25.9 | ~0 | **56.4 GiB** | fits |
  | fp32 + adamw | 30.5 | 25.9 | 51.8 | **108.1 GiB** | over by 13.2 |
  | bf16-compute + adafactor | 15.2 | 51.8 | ~0 | **67.0 GiB** | fits |
  | bf16-compute + adamw | 15.2 | 51.8 | 51.8 | **118.8 GiB** | over by 23.8 |
  | bf16-compute + adamw-8bit | 15.2 | 51.8 | 12.9 | **80.0 GiB** | fits, ~15 GiB left |

  8-bit AdamW is the only AdamW that fits, and it costs 12.9 GiB to buy state that
  Adafactor provides in 9 MB — while discarding the one hyperparameter this project has
  actually measured (lr 5e-4 under Adafactor). Not worth it for a run whose purpose is to
  test two data-side interventions.
- **TF32 (`--matmul-precision high`) is free speed that is bit-exact where it counts —
  but the gain shrinks with the window.** 1.70x at 8064 (11.8 vs 20.1 s), only **1.38x at
  32768** (90.1 vs 124.2 s), at identical memory in both cases. torch defaults to
  `highest`, so every plain-fp32 number above is a *true*-fp32 matmul. The narrowing is
  the point: TF32 accelerates the linear layers, and at a long window attention — served
  in fp32 by the memory-efficient kernel, which TF32 does not speed up the same way —
  takes a growing share of the step. It reduces only the matmul's internal accumulation to
  10 mantissa bits; the latents, the TWN threshold, `ternarize_group` and its deliberate
  fp16 scale rounding are all elementwise fp32 and untouched. bf16 remains 3.6x ahead of
  it at 32768, because bf16 reaches FlashAttention and TF32 cannot.
- **`--matmul-precision medium` is not worth considering.** 90.8 s at 32768 against
  `high`'s 90.1 — identical within noise, for two fewer mantissa bits. That null result is
  the confirmation of the paragraph above: at a long window what is left in the step is
  attention, not linear-layer matmul precision, so there is nothing for `medium` to buy.
  `high` is the only fp32 precision knob worth having.

#### Does reduced precision actually move the ternary codes?

The obvious worry about `--compute-dtype bf16` is that it rounds the latent to 8 mantissa
bits *before* ternarizing, so a weight near the TWN threshold ternarizes differently than
the fp32 master the export reads. **Measured, it does not happen at all:**

| tensor | codes differing, fp32 vs bf16 ternarization |
|---|---|
| `layers.17.self_attn.q_proj` | 0 / 16,777,216 |
| `layers.35.mlp.gate_proj` | 0 / 50,331,648 |
| `layers.0.mlp.down_proj` | 0 / 50,331,648 |

Nor is anything close: counting weights within a given relative distance of `delta`, on
real trained latents, gives **0 within bf16 precision, 0 within TF32 precision, and 0
within *fp32* precision**. The reason is structural — a ternary latent sits at 0 or `±s`
while `delta = 0.7·mean|W|` sits between them, so the boundary region is empty by
construction. Codes only cross it transiently, while flipping.

What bf16 *does* move is smaller and elsewhere: `ternarize_group` rounds the scale to
**fp16 on purpose**, to match deployed Q2_0 numerics, and under bf16 compute that scale is
bf16-rounded on top — measured 0.05-0.10% off the value the exported GGUF will carry. And
the gradients are computed at 8 mantissa bits, which shifts `gnorm` and therefore what
`GradSpikeGuard` treats as a spike. Neither is settled by argument; §10.6 measures them.

### 10.3 What the port changed in the code

`qat/_device.py` is new and owns every backend difference; the trainer asks it rather than
branching on `dev ==`. The line it replaces was
`dev = "mps" if torch.backends.mps.is_available() else "cpu"`, which on a CUDA box
**silently selects CPU** — no error, ~100x slower, checkpoints still saving.

| decision | MPS | CUDA | why it inverts |
|---|---|---|---|
| `foreach` | False | True | MPS multi-tensor kernels deadlock at full-model scale; on CUDA they replace ~250 small launches |
| chunked SDPA | forced | off | MPSGraph INT_MAX score cap vs a fused kernel (given the fp32 GQA fix) |
| `max_window` w/o chunking | 8191 | none | `n_heads * S^2 < 2^31` is a Metal limit |
| teacher dtype | fp16 | bf16 | no bf16 on M1-generation parts |
| `--empty-cache-every` | 5 | 0 (off) | macOS OOM-kills a run whose working set creeps into swap; CUDA has no such mode and the release costs a sync plus reusable blocks |
| memory reporting | `current_allocated_memory` | `max_memory_allocated` | MPS exposes no peak counter |

Also changed:

- **`metrics.jsonl` gains `mem_peak_gib` and `device`**, and the step line prints
  `mem=<live>/<peak>GiB`. The peak is what a run is sized from — the transient a
  checkpoint save or a long-window validation spikes to is what decides whether the next
  one OOMs, and a point sample of live bytes will not show it.
  `scripts/qat_progress_report.py` renders both.
- **`--matmul-precision {highest,high,medium}`** (new). This is a different knob from
  `--compute-dtype`: the latents, the TWN threshold and `ternarize_group` are all
  elementwise fp32 and stay bit-exact, so the codes a step produces are unperturbed in a
  way `--compute-dtype bf16` cannot promise. Only the matmul accumulation is reduced
  (TF32 keeps 10 mantissa bits, bf16 8). Default `highest` = true fp32 = what every
  published run used. Note torch defaults to `highest`, so **the fp32 numbers above are
  true-fp32 matmuls, not TF32.**
- **Flip telemetry now reads the fp32 masters under bf16 compute.** `snapshot_codes` read
  `linear.weight` — the live *bf16 copy* — while `export_qat` ternarizes the masters. bf16
  carries 8 mantissa bits, so a latent within ~0.2% of the TWN threshold ternarizes
  differently in the two, and flips get recorded at the wrong step. Since flip velocity,
  not loss, is how this project decides whether a ternary run is learning at all, that
  instrument has to read the tensor that will actually ship. `latent_weights(model, opt)`
  resolves it; unit-tested.
- **`scripts/probe_window_budget.py` refuses to start against a busy card.** A killed
  sweep leaves its trainer holding the whole GPU, the next configuration cannot even load
  the model, and the probe records "OOM" against a configuration that never ran. This
  happened here: a leaked 32768 run sat at 96.9 of 97.9 GiB and two innocent configs were
  recorded as OOM. It is the CUDA analogue of §5's macOS cooldown trap and fails the same
  misleading way. Swap reporting also now reads `/proc/meminfo` on Linux.
- **`scripts/watch_qat_run_cuda.sh`** is the CUDA sibling of `watch_qat_run.sh` (added,
  not edited — §"Constraints"). It watches VRAM, the process list on the card, and the log
  for real tracebacks, because on CUDA an OOM is an ordinary exception rather than §5's
  silent SIGKILL.
- **`export_qat` now raises if `chat_template.jinja` is missing** instead of skipping it
  silently. Without the template the F16 conversion bakes a thinking-enabled Qwen3 default
  and the exported model emits `<think>` and never tool-calls — which reads as a model
  that lost its agentic ability, not as a missing file.

Unit tests: **1003 pass** (915 before; the additions cover the backend table, the fp32 GQA
predicate, and the bf16 telemetry path). `mypy src` reports the same 5 errors in
`qat/` before and after the port — none introduced.

### 10.4 Environment gaps found on a fresh box

None of these are in any runbook, and all three block a step:

1. **`out/exp-057/chat_template.jinja`** — `qat/corpus.py` reads it from `exp-057/`, but
   the unpacked HF repo ships it inside `model/`. Copy it up one level or every corpus
   build dies on `FileNotFoundError`.
2. **`vendor/llama.cpp-prism` is not a tracked submodule** and seven scripts export the
   path without ever saying where it comes from. It is
   `https://github.com/PrismML-Eng/llama.cpp` (`prism` branch, default), recorded only in
   the `prism-ml/Ternary-Bonsai-8B-gguf` model card. Build command now in
   `docs/ternary_qat.md`. Confirmed: `llama-quantize` lists `41 or Q2_0`.
3. **`prism-ml/Ternary-Bonsai-8B` 404s.** The weights live at `…-8B-gguf` (packed) and
   `…-8B-unpacked` (the trainable fp16 safetensors `out/exp-057/model` is a snapshot of).

**The SWE-rebench eval cannot run on this box.** It is an unprivileged container: no
Docker daemon, no `/var/run/docker.sock`, and no `cap_sys_admin`, so no container runtime
can be installed. Training and export work; the A/B has to be graded on a Docker-capable
machine. The holdout *file* is network-only and was built here.

### 10.5 Resuming this run with more data

The point of this branch is that the next iteration should be a corpus change and a
launch, not a re-derivation. Everything below is already in place.

**What is pinned.** The SFT blob is `out/corpora/qwen3-universal-v2/sft.jsonl.gz`
(sha256 `32ed736c…`, 22,686,353 bytes). The packed corpora are
`out/exp-058/sft_corpus_{universal,val}_32768.pt`, and the trainer stores a
`corpus_fingerprint` in every checkpoint — a `--resume` against a corpus that does not
match is refused rather than silently continuing on different data. **Adding data
therefore ends the old run**; it does not extend it.

**To rebuild with more data:**

```bash
# 1. regenerate the SFT blob (data.universal writes sft.jsonl.gz beside the corpus)
PYTHONPATH=src .venv/bin/python scripts/build_universal_corpus.py --ctx 32768 ...

# 2. repack, at the window the run will use. --max-tool-tokens scales WITH the window:
#    1024 drops 28% of all conversation content and was only ever right at 4096.
PYTHONPATH=src .venv/bin/python scripts/build_sft_qat_corpus.py \
    --window 32768 --max-tool-tokens 4096 --min-density 0.05 \
    --out out/exp-058/sft_corpus_universal_32768.pt
# ...and again with --split test for the val corpus.

# 3. relaunch (fresh --out; do not resume across a corpus change)
bash scripts/run_sft32k_qat_cuda.sh <tag> <lr> <epochs>
```

**Do not plan to extend epochs with `--resume`.** `total_steps` is recomputed from
`--epochs`, so resuming a finished 1.0-epoch run at 2.0 puts the cosine schedule back
near its peak with no warmup: **5.0e-5 -> 2.9e-4, a 5.9x step up**, at exactly the
annealed point. That is the same shape as the trigger that cost sft8k-full 90 steps.
Commit to the epoch count at launch. `--resume` is for continuing an *interrupted* run
of the same schedule on the same corpus, which is what it is safe for.

**What scales with more data, and what does not.** Steps per epoch is
`n_windows / grad_accum`, so a corpus 2x the size doubles the run at a fixed
tokens-per-step — the numbers in §10.2 are per-step and stay valid. What does *not*
carry over is the stop-token ratio: `--stop-weight` corrects the measured imbalance in
*this* blob (35,359 terminating `<|im_end|>` in 6,071,948 targets, one per 172). Recheck
it against the new corpus's `im_end_targets` and rescale, or the weight silently means
something different.

**If the window changes**, re-run the probe rather than interpolating §10.2 — fp32 cost
per trained token is *not* flat on CUDA (+55% from 8064 to 32768), unlike the MPS
plateau §3 recorded.

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_window_budget.py \
    --window W --trained-tail 0 --steps 3 --grad-accum 1 --cooldown 0
```

**Probe caveat — the tables in §10.2 are WARM-UP steps.** `train.py` emits a step record
at step 1 and then every 5th, so a `--steps 3` probe only ever logged **step 1**, which
carries kernel autotune and allocator growth. Those numbers are comparable *to each other*
(every config paid the same cost) but they are not the rate a 613-step run bills at. To
price a real run, run ~12 steps and take the marginal rate between the step-5 and step-10
`elapsed_s` in `metrics.jsonl` — that is what §10.6 does.

### 10.6 Pre-flight on the real corpus

*(populated by `preflight2` — real corpus, real config, 12 steps, validation and a
checkpoint save included, fp32 vs TF32 vs bf16 on an identical window order.)*

### 10.7 Grading the A/B somewhere else

The eval cannot run on the training box (§10.4). What has to move is small:

| artifact | what it is |
|---|---|
| `out/exp-057/Ternary-Bonsai-8B-<tag>-Q2_0.gguf` | the run's export, ~2.5 GiB |
| `out/exp-057/Ternary-Bonsai-8B-vanilla-Q2_0.gguf` | the untrained control, same build |
| `out/external/swe-rebench/holdout50.jsonl` | 50 instances, **0 overlap** with the 71 trained on |

Both GGUFs must come from the *same* llama.cpp build, and the eval box needs a Docker
daemon plus the `swebench` extra:

```bash
PYTHONPATH=src python scripts/run_swebench_eval.py \
    --models Ternary-Bonsai-8B-vanilla-Q2_0.gguf Ternary-Bonsai-8B-<tag>-Q2_0.gguf \
    --holdout holdout50.jsonl --workspace out/ab-<tag>
```

**Loop fraction is the primary endpoint** for this iteration, not resolve rate: sft8k-full
already showed that behaviour can move (tool errors 0.65 -> 0.33) while capability does
not (0/10 both ways). Report resolved, patch rate, tool-error rate, `max_turns` exits and
loop fraction together. At 0/50 the 95% upper bound on the resolve rate is ~5.8%, versus
~31% at n=10 — which is the whole reason for the larger holdout.
