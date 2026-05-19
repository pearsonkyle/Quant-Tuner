# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`quant-tuner` takes a HuggingFace model plus a corpus derived from real prompt/response
logs and produces a GGUF quantization calibrated to that distribution. It then benchmarks
the result against an FP16 reference (KLD, perplexity, prefill/decode tok/s) and against
task-level metrics (tool-call accuracy on held-out sessions).

Status: pre-alpha. The Python API is the working surface; the `quant-tuner` CLI is mostly stubs
(only `leaderboard` is wired up). The `experiments/` scripts in `scripts/` are the
authoritative entry points used to produce the published results.

## Setup and common commands

```bash
# One-time: fetch + build vendored llama.cpp (pinned commit 45b455e6)
git submodule update --init --recursive
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON   # Linux+CUDA: -DGGML_CUDA=ON
cmake --build vendor/llama.cpp/build -j

# Python env
uv sync

# Tests / lint / types
uv run pytest                                   # all unit tests
uv run pytest tests/unit/test_imatrix.py        # single file
uv run pytest tests/unit/test_imatrix.py::test_name
uv run ruff check src tests
uv run mypy src

# Leaderboard CLI (the one non-stub command)
uv run quant-tuner leaderboard --results out/<run>/results.csv --out LEADERBOARD.md
```

Override the llama.cpp build location with `LLAMA_CPP_DIR=/path/to/llama.cpp` if not
using the vendored submodule. `paths.llama_bin(name)` resolves binaries from
`$LLAMA_CPP_DIR/build/bin/`.

## Architecture

### Pipeline
```
HF model + usage-log JSONL
   → data.ingest + data.split   (train.txt / test.txt / holdout.jsonl)
   → quantize.convert.hf_to_f16_gguf  (HF → F16 GGUF, one-time)
   → calibrate.{imatrix,awq,gptq}     (produce calibration artifact)
   → quantize.gguf.quantize(... , imatrix=...)  (F16 → Q* GGUF via llama-quantize)
   → bench.runner.bench_one           (BPW + KLD + PPL + llama-bench speed → CSV row)
   → leaderboard.aggregate            (CSV → markdown with SQS scoring)
```

### Three calibration methods (all produce a standard GGUF; no runtime cost)
- **imatrix** (`calibrate/imatrix.py`): consumes a base `imatrix.gguf` produced by
  `llama-imatrix` (via `models.llama_cpp.imatrix`), then re-weights per-tensor using one of
  several **variants**:
  - `analytic`, `mix_50`, `hybrid_custom` — closed-form, no model load. `hybrid_custom`
    is the published winner; it combines `E[a²]` with `‖W[:,c]‖²·E[a²]` per-tensor.
  - `outlier_l4`, `outlier_max` — require an HF forward pass to capture heavy-tail stats.
  - SSM tensors (Mamba etc.) always pass through with raw `E[a²]` — see
    `models.hf_gguf_map.is_ssm`; output-aware re-ranking is invalid for them.
- **awq** (`calibrate/awq.py`): activation-aware scaling folded into RMSNorm.
- **gptq** (`calibrate/gptq.py`): Hessian-based rounding with error compensation; has a
  `verify_perplexity` guardrail.

### Tensor naming
`models/hf_gguf_map.py` maps HF parameter names to GGUF tensor names. Anything that
crosses the HF↔GGUF boundary (imatrix variants, AWQ apply) goes through this mapping.

### Bench
`bench/runner.py` defines `BenchRow` and `CSV_COLUMNS`. Sub-modules:
- `bpw.py` — bits-per-weight from `n_params(f16)` and file size.
- `kld.py` — `build_baseline(f16, eval_ds)` produces a reference KLD file via llama.cpp;
  subsequent runs diff against it.
- `speed.py` — wraps `llama-bench` for prefill/decode tok/s + TTFT with N repetitions
  (mean ± stdev).

### Experiment scripts (`scripts/`)
These are the real entry points; treat them as runnable specs of what a full run looks like:
- `run_omnicoder_q4_k_m.py`, `run_omnicoder_wiki_vs_custom.py`,
  `run_omnicoder_mixed_corpus.py` — calibration ablations on Tesslate/OmniCoder-9B.
- `build_toolcall_holdout.py` — extracts the 25-session tool-call holdout from
  `logtrain.jsonl` (`test` + `holdout` slices, disjoint from `train` used for calibration).
- `eval_toolcall.py`, `run_toolcall_all.py`, `run_toolcall_reps.py` — task-level eval
  via OpenAI-compatible llama.cpp server. Sampling params (`T`, `top_p`, `top_k`, `min_p`,
  `presence`, `repeat_penalty`, seed) are passed via `extra_body`. Runner reuses one
  server per model across N repetitions; aggregator emits mean ± stdev.
- `rebench_speed.py` — re-runs speed bench only (since decode tok/s drifts with thermal
  state across long bench sessions — see README caveat).

### Workspace layout
`paths.Workspace(root)` is the canonical per-run output directory; `workspace.ensure()`
creates `model_extracted/`, `corpus/`, `calibration/`, `gguf/`, `eval/` and reserves
`results.csv` at the root. Experiment scripts write to `out/<run-name>/` following this layout.

### Recipes
`src/quant_tuner/recipes/*.yaml` declare end-to-end recipes (`q4_k_m_imatrix`,
`q4_k_m_awq`, `q4_k_m_gptq`, `q4_k_m_none`). These are reference configs; the
recipe-runner is not yet wired to the CLI.

## Conventions worth knowing

- GGUF linear weights are stored `[n_out, n_in]`. Summing `W²` over axis 0 gives
  `‖W[:, c]‖²` per input channel — used throughout `calibrate/imatrix.py`.
- `_load_base_imatrix` divides `*.in_sum2` by `*.counts` to recover `E[a²]`; new variants
  should preserve this normalization.
- Speed numbers have a known thermal artifact: rows that run later in a session drift
  lower. SQS (which weights speed equally with compression) is noisier than KLD; for
  "which imatrix is best?" read **KLD and tool-call** columns.
- Calibration `train`, eval `test`, and `holdout` slices come from the same source
  (`logtrain.jsonl`) but are disjoint — preserve this invariant when adding new evals.
