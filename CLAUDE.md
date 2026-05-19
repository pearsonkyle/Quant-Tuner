# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`quant-tuner` takes a HuggingFace model plus a corpus derived from real prompt/response
logs and produces a GGUF quantization calibrated to that distribution. It then benchmarks
the result against an FP16 reference (KLD, perplexity, prefill/decode tok/s) and against
task-level metrics (tool-call accuracy on held-out sessions).

Status: beta. The `quant-tuner` CLI is recipe-driven and runs end-to-end
(`run`, `bench`, `leaderboard` all wired). The orchestrator
`scripts/reproduce_leaderboard.py` chains the OmniCoder study's stages and is
the canonical entry point for reproducing the published table.

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

# CLI (recipe-driven)
uv run quant-tuner run --recipe q4_k_m_imatrix --model org/repo --logs logs.jsonl --workspace out/run
uv run quant-tuner run --recipe q4_k_m_imatrix --model X --logs Y --workspace W --dry-run  # validate-only
uv run quant-tuner bench --quant Q.gguf --reference F16.gguf --eval EVAL.txt --out results.csv
uv run quant-tuner leaderboard --results out/<run>/results.csv --out LEADERBOARD.md
```

Override the llama.cpp build location with `LLAMA_CPP_DIR=/path/to/llama.cpp` if not
using the vendored submodule. `paths.llama_bin(name)` resolves binaries from
`$LLAMA_CPP_DIR/build/bin/`.

## Architecture

### Pipeline
`pipeline.run_pipeline(RunConfig)` is the canonical end-to-end flow; `cli.run`
is a thin shim over it. Stages (each idempotent via `experiments.step`):

```
HF model + usage-log JSONL
   → pipeline.extract_and_convert      (HF → F16 GGUF, one-time)
   → pipeline.prepare_corpora          (data.ingest + data.split + stratified_pack)
   → pipeline.calibrate                (method-dispatched: imatrix/awq/gptq/none)
   → pipeline.quantize_model           (F16 → Q* GGUF via llama-quantize)
   → pipeline.bench                    (BPW + KLD + PPL + llama-bench → CSV row)
   → leaderboard.aggregate             (CSV → markdown with SQS scoring; separate CLI step)
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

### Tool-call evaluation (`eval/`)
Task-level eval lives in `src/quant_tuner/eval/`:
- `eval/scoring.py` — pure type-aware comparators (`compare_value`, `param_score`,
  `is_schema_valid`). Fully unit-tested.
- `eval/server.py` — `running_server(model_path)` context manager spawns/health-checks
  /tears-down `llama-server` via `paths.llama_bin`.
- `eval/toolcall.py` — `Sampling` dataclass, `eval_per_turn`/`eval_rollout`, and the
  top-level `run_toolcall_eval(holdout, model_path=… | base_url=…)` orchestrator
  returning an `EvalSummary` dataclass. Sampling params (`T`, `top_p`, `top_k`, `min_p`,
  `presence`, `repeat_penalty`, `seed`) are passed via `extra_body`; the OpenAI SDK
  forwards them unmodified.

### Experiment scripts (`scripts/`)
The OmniCoder reproduction is here; the CLI handles ad-hoc runs.
- `reproduce_leaderboard.py` — orchestrator chaining 7 stages (extract → 3 calibration
  stages → holdout → speed rebench → tool-call reps → render). Each subprocess-isolated.
- `run_omnicoder_{q4_k_m,wiki_vs_custom,mixed_corpus}.py` — the three calibration stages,
  using `experiments.step()` for idempotency.
- `build_toolcall_holdout.py` — samples the 25-session tool-call holdout from the
  `test + holdout` slices of `logtrain.jsonl`.
- `eval_toolcall.py` — thin argparse CLI over `eval.run_toolcall_eval`. Pass `--base-url`
  to reuse a server across calls.
- `run_toolcall_all.py`, `run_toolcall_reps.py` — per-model server reuse and multi-rep
  aggregation (mean ± stdev across N reps).
- `rebench_speed.py` — re-runs speed bench only (decode tok/s drifts with thermal state
  across long sessions — see README caveat).

### Workspace layout
`paths.Workspace(root)` is the canonical per-run output directory; `workspace.ensure()`
creates `model_extracted/`, `corpus/`, `calibration/`, `gguf/`, `eval/` and reserves
`results.csv` at the root. Experiment scripts write to `out/<run-name>/` following this layout.

### Recipes
`src/quant_tuner/recipes/*.yaml` declare end-to-end recipes (`q4_k_m_imatrix`,
`q4_k_m_awq`, `q4_k_m_gptq`, `q4_k_m_none`). Loaded via `RunConfig.from_yaml`
and executed by `pipeline.run_pipeline`. `cli._resolve_recipe` accepts both
bare names and absolute paths; `PLACEHOLDER` fields are rejected with a clear
"pass `--model` / `--logs`" error.

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
