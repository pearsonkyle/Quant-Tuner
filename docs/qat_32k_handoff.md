# 32K ternary-QAT: state of play and how to resume elsewhere

Written 2026-08-14 after the work below crashed the M4 Max it was running on (see
"The crash" — it was an operator error, not a property of the code). Everything here is
committed; nothing is left in a working tree.

Target: `prism-ml/Ternary-Bonsai-8B` continued QAT, all-36 layers, fp32 latents, Adafactor.
Reference box: **M4 Max, 128 GB unified memory, macOS/MPS, torch 2.12, transformers 5.8.1.**
Every number below is from that box; **none of it transfers to CUDA unchanged** (see
"Porting to a different machine").

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
