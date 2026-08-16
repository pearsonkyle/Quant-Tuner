# Prompt for a new session: ternary QAT with more data, on CUDA

Ship **one file** alongside this prompt: `out/corpora/qwen3-universal-v2/sft.jsonl.gz`
(22 MB). Everything else is rebuilt from it. Paste the text below the `---` into a fresh
clone of this repo, branch `claude/sft-split-fix-and-mixture`.

**Upload `-v2`, not `out/corpora/qwen3-universal/`.** The older directory predates commit
`7a50385`, which fixed the SFT export splitting `broad-instruct` on `half` (the
*quantization* axis) instead of the eval boundary. Pre-fix, `--split train` silently
withheld 2,554 of the 5,536 supplement rows — precisely the wrong artifact for a run whose
purpose is more data. `-v2` has train 6,170; the old one has 3,616.

---

I want to plan and launch the next Ternary-Bonsai-8B continued-QAT run, this time on **all
five SFT sources** instead of the 12 SWE trajectories the last runs used, and **on a CUDA
box**. All prior work was done on an M4 Max / MPS, so a large part of this job is porting.

Read `docs/qat_32k_handoff.md` and `docs/ternary_qat.md` first. The long-window work is
done, validated and committed — do not redo it. But **treat every number in those docs as
MPS-specific and every code path as MPS-shaped until you have checked it.**

## Start here: the trainer has no CUDA branch

`src/quant_tuner/qat/train.py:450` is literally:

```python
dev = "mps" if torch.backends.mps.is_available() else "cpu"
```

On a CUDA box that silently selects **CPU** and the run is unusable — it will not error, it
will just be ~100x too slow. This is the first thing to fix, and it is not the only MPS
assumption in the file. Audit at minimum:

| location | what it assumes | what CUDA wants |
|---|---|---|
| `train.py:450` | mps-or-cpu | add a `cuda` branch |
| `:529`, `:693` | `foreach=False` on AdamW and `clip_grad_norm_` | an MPS deadlock workaround; on CUDA `foreach=True` is a real speedup — flip it, don't inherit it |
| `:467` | chunked SDPA forced on when `dev == "mps"` | CUDA has fused SDPA/FlashAttention that never materializes the score matrix, so `chunked_causal_sdpa` is probably **unnecessary and slower**. Benchmark `--no-chunked-attention` against the default. Note `--trained-tail` still requires the patched path on every device (it carries the prefix K/V) — but you likely don't want `--trained-tail` at all, see below |
| `:508` | teacher fp16 on mps, fp32 elsewhere | fine, but bf16 is the natural CUDA choice |
| `:607,638,677,787,815` | `torch.mps.empty_cache()` | `torch.cuda.empty_cache()`; the every-5-steps cadence was tuned against macOS swap and is likely over-eager |
| `:790` | `torch.mps.current_allocated_memory()` | `torch.cuda.max_memory_allocated()`, else the reports show 0 GiB |

Please make these changes cleanly (a small device-abstraction rather than a widening pile
of `if dev ==` branches), keep the MPS path working, and keep the unit tests green
(`uv run pytest`; 915 tests pass today).

**Also re-test, don't inherit, these three MPS-driven decisions:**

- **`--optim adafactor`** was chosen because AdamW's ~55.6 GB of state did not fit in 128 GB
  of *shared* memory. On an 80 GB+ dedicated GPU, revisit AdamW — it is the better optimizer
  and the memory objection may not apply.
- **`--compute-dtype bf16`** is a pessimization on Metal (68,346 ms vs 2,456 ms per layer at
  32768 — a pathological MPS path, not a gradual falloff). On CUDA bf16 with fp32 masters is
  the normal choice and should be materially faster. Note `--dtype bf16` (bf16 *latents*) is
  still wrong everywhere: bf16 underflows the ternary threshold and no codes flip.
- **`--trained-tail` / prefix-context.** It is bit-exact and validated, but it *discards
  training signal* (~T/W of targets) and only exists to get past an activation ceiling.
  With FlashAttention that ceiling likely disappears. Default to full-gradient.

Tell me what you changed and what you measured. If it turns out CUDA reaches 32768
full-gradient with room to spare, say what the new ceiling is.

## The uploaded corpus

`sft.jsonl.gz` (put it at `out/corpora/qwen3-universal-v2/sft.jsonl.gz`) is what
`data.universal` emits: **6,643 full conversations**, untokenized, with real tool schemas
and scrubbed system prompts.

| field | |
|---|---|
| rows | 6,643 |
| sources | `broad-instruct` 5536, `logs-agents` 435, `redteam-refusals` 348, `logs` 253, `swe-trajectories` 71 |
| splits | `train` **6,170**, `holdout` 406, `test` **67** |
| schema | `id, source, split, messages, tools, n_messages, n_tool_calls, n_tool_results, n_reasoning, n_chars, system_scrub` |

`holdout`/`test` are held out of `--split train` by default — keep it that way. The
`broad-instruct` holdout is exactly 278 rows (209 eval + 69 val), the rows
`corpus.eval.broad.txt` is built from, so the PPL/KLD eval slice stays genuinely held out.

Note `test` is only 67 rows, which is why the previous validation corpus was
`logs`/`logs-agents` distribution with **no** held-out SWE. Decide whether that is good
enough or whether the val slice needs rebuilding.

`scripts/export_sft_chat_jsonl.py` re-exports this with every row rendered through a real
chat template first, so a row that would blow up mid-training surfaces at export time
instead. Worth running once on the new box before committing to a long run —
`--on-error skip` reports the failures rather than dying.

Build the tensors with `scripts/build_sft_qat_corpus.py`:

```bash
# training corpus — all sources, `train` split
PYTHONPATH=src python scripts/build_sft_qat_corpus.py \
    --sft out/corpora/qwen3-universal-v2/sft.jsonl.gz \
    --window 32768 --max-tool-tokens 8192 --min-density 0.05 \
    --out out/exp-058/sft_corpus_universal_32768.pt

# validation corpus — the `test` split
PYTHONPATH=src python scripts/build_sft_qat_corpus.py \
    --sft out/corpora/qwen3-universal-v2/sft.jsonl.gz --split test \
    --window 32768 --max-tool-tokens 8192 --min-density 0.05 \
    --out out/exp-058/sft_corpus_val_32768.pt
```

`scripts/export_qat_corpus_jsonl.py` converts a built `.pt` to `jsonl.gz` (same windows, no
torch needed) if you want to hand the packed corpus to a different training framework.

`--max-tool-tokens` scales with the window (3072 at 8064, 4096 at 12288, 8192 at 32768) —
at 1024 it drops 28% of all conversation content. `--source` restricts to a subset;
`--budget SOURCE=TOKENS` caps one source. (Its `--window` help text still claims "8064
remains the largest full-gradient window that trains clean" — **stale**, the score-recompute
fix superseded it. Fix the docstring if you touch the file.)

**Reproduce these as a check that your rebuild matches mine:**

| build | windows | tokens | labeled targets | `<\|im_end\|>` targets |
|---|---|---|---|---|
| universal @ 8064 | 2138 | 17.24 M | 6,060,840 | 35,046 |
| universal @ 32768 | 613 | 20.09 M | 6,071,948 | 35,359 |
| val (`test`) @ 32768 | 81 | 2.65 M | 684,761 | 3,513 |
| SWE-only @ 32768 (last run's scope) | 26 | 0.85 M | 267,404 | 1,915 |

Both universal rows carry `broad-instruct: conversations_used 5258` in the blob's
`per_source` — that is the marker of a post-fix build. If yours says **2704** you built from
the old corpus directory and are missing 2,554 conversations. I have a 16128 blob locally
but it is pre-fix, so its numbers are deliberately omitted here rather than shipped as a
target you cannot hit.

## Two facts that should shape the plan

1. **Target count is ~6.0 M at every window size.** Window length trades *context* for
   *wall-clock*, not training signal — sessions pack contiguously, so nothing is lost at
   8064. What a longer window buys is conditioning a trajectory's tail on its start (27% →
   68% → 97% of SWE conversations whole at 8064 / 16128 / 32768).
2. **On the M4 Max, one epoch of the universal corpus cost ~53 h at 8064, ~98 h at 16128,
   ~152 h at 32768.** Those are reference numbers for the *shape* of the tradeoff only —
   re-derive them on your GPU, where they should be dramatically lower and where the
   window/cost curve may not have the same slope at all (FlashAttention changes the
   attention term). The point that survives: "more data" here is a **budget-allocation**
   problem, not a build problem.

## What I want from you

1. **Port, then measure.** Fix the device handling, then get real s/step on this hardware
   (`scripts/probe_window_budget.py --window W --trained-tail 0 --steps 2 --grad-accum 2`)
   across at least 8064 / 16128 / 32768, with and without chunked attention, fp32 vs
   `--compute-dtype bf16`. Report GPU model, VRAM, and count. That table replaces the MPS
   one and belongs in `docs/qat_32k_handoff.md`.
2. **Propose a concrete recipe** — window, epochs (fractional is fine and expected),
   grad-accum, `--stop-weight`, `--val-windows`, LR, optimizer — with implied wall-clock and
   the reasoning for each choice. Reuse what prior work settled where it is not
   device-specific: **lr 5e-4 for ~2.2 epochs is the measured sweet spot** (3e-4 flips ~0%
   of codes, 8 epochs memorizes), fp32 latents, all-36 layers.
3. **Take a position on the source mixture.** The last run was SWE-only and moved
   *behaviour* (tool-error rate 0.65 → 0.33) but not *capability* (0/10 resolved,
   `max_turns` on 7/10, loops on 97% of trajectories). `broad-instruct` is 83% of rows but a
   small share of tokens; `swe-trajectories` is 71 rows carrying the task we care about.
   Uniform, upweighted, or something else? `build_sft_qat_corpus.py --budget SOURCE=TOKENS`
   bakes a mixture into the blob — say whether that suffices or the trainer needs a sampling
   weight.
4. **Report as it runs.** I've added `scripts/qat_progress_report.py` — it reads
   `metrics.jsonl` only, so it is device-agnostic and works on a finished run too:

   ```bash
   python scripts/qat_progress_report.py out/exp-058/<run> --watch 1800
   ```

   It writes `report.md` (rewritten each interval) and appends to `report_history.md`.
   **Run it for the duration and check in on it rather than polling the process.** I care
   most about two of its sections: the **flip velocity** table (`flip_pct_delta` — a ternary
   model only learns by flipping codes; the loss falls on scale drift alone, so a falling
   loss with ~0% flips means it is not training) and the **loss-by-source** table, which is
   the evidence for whether the mixture choice in (3) was right. Extend the script if
   something is missing, and give me a short written read of it at least once a day —
   what the flips are doing, whether any source is flat, and whether the ETA has moved.
5. **Say how we will know it worked.** The SWE-rebench holdout is n=10; at 0/10 both before
   and after, the 95% upper bound on the true resolve rate is ~31%, so it cannot settle
   this. Either enlarge the holdout first (`scripts/build_swebench_holdout.py`, kept
   `--exclude`-disjoint from the training pool) or name a proxy metric you trust. I would
   rather spend two hours on the measurement than 50 on a run I can't read.

## Constraints

- **Read the flip telemetry and `gnorm`, not the loss.** Stated three times because every
  previous iteration of this work got misled by a nice-looking loss curve.
- **Do not launch a multi-day run until I have seen the recipe and the wall-clock estimate.**
- Ignore the macOS-specific operational advice in the docs — `sysctl vm.swapusage`,
  `scripts/watch_qat_run.sh`, the "let the box settle before a model load" rule, and the
  "OOM kills give no traceback" warning are all unified-memory/macOS artifacts. On CUDA an
  OOM raises `torch.cuda.OutOfMemoryError` with a real traceback; use `nvidia-smi` and
  `torch.cuda.max_memory_allocated()`. If you write a CUDA equivalent of the watch script,
  add it rather than editing the macOS one.
- Docker cleanup, if any: `scripts/docker_housekeep.sh` only (SWE images + dangling, never
  `-a`).

Start with the port and the measurement.
