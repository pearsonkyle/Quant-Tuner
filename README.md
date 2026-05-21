# quant-tuner

Calibrate your own GGUF quantizations from real usage logs.

`quant-tuner` takes a HuggingFace model and a corpus derived from your own
prompt/response logs, then produces a GGUF quantization tuned to the
distribution your model actually sees in production. It also benchmarks
the result against an FP16 reference using KL-divergence, perplexity, and
prefill/decode speed.

## Methods

| Method   | What it does                                            | Pre-quant cost | Adds inference cost? |
| -------- | ------------------------------------------------------- | -------------- | -------------------- |
| imatrix  | Per-tensor importance from a forward pass               | Low            | No                   |
| AWQ      | Activation-aware weight scaling folded into RMSNorm     | Medium         | No                   |
| GPTQ     | Hessian-based weight rounding with error compensation   | High           | No                   |

All three produce a standard GGUF file that runs on any unmodified `llama.cpp`.
None of them adds runtime cost — calibration changes *which* weights get the
quantizer's budget, not the inference path.

See `docs/methods.md` for the algorithmic details and `docs/benchmarks.md` for
the metric definitions.

## Results so far (Tesslate/OmniCoder-9B @ Q4_K_M)

End-to-end run on a real 9B coding model. Eight rows comparing two **imatrix
techniques** (`stock` = llama.cpp's standard `E[a²]`, `hybrid` = output-aware
`max(L1-norm(E[a²]), L1-norm(‖W[:,c]‖²·E[a²]))`) × three **calibration corpora**
(`custom` = `logtrain.jsonl`, `wiki` = WikiText-2 test, `mixed` = wiki + 250k
tokens of logtrain), plus the uncalibrated Q4 floor and the F16 ceiling.

| | Stock imatrix | Hybrid imatrix |
| --- | --- | --- |
| **custom** (logtrain only) | row | row |
| **wiki** (WikiText-2) | row | row |
| **mixed** (wiki + 250k logtrain) | row | row |

Eval data: KLD on a held-out 48k-token test split; tool-call on a disjoint
25-session holdout (9 claude + 16 qwen) drawn from `test + holdout` slices of
`logtrain.jsonl` (both splits are disjoint from the train slice used for
calibration).

Sorted by Mean KLD (lower = closer to F16; the speed columns also vary, but
read those with the caveat below):

| Model | Size | Mean KLD | Same Top p | Tool Sel % | Param Acc % | Schema % | Rollout % | MMLU-Pro % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline/fp16        | 
| hybrid / mixed       | 
| **hybrid / custom**  | 
| stock / custom       | 
| stock / wiki         | 
| baseline/Q4_K_M-none | 

**Reproduce the whole table:**
```bash
uv run python scripts/reproduce_leaderboard.py            # ~17 h cold, ~5 min if cached
uv run python scripts/reproduce_leaderboard.py --quick-toolcall  # ~6 h, 2 reps × 8 instead of 10 × 8
uv run python scripts/reproduce_leaderboard.py --skip-toolcall   # skip the 14-h tool-call stage
```

`reproduce_leaderboard.py` chains seven stages (extract → F16 → calibrate × 3 corpora → holdout → speed rebench → tool-call reps → render). Every stage is idempotent — re-running on a populated workspace just verifies state and re-renders `LEADERBOARD.md`. Individual stages live in `scripts/run_omnicoder_*.py`, `scripts/rebench_speed.py`, and `scripts/run_toolcall_reps.py`.

**For any other model**, use the parameterized reproducer:

```bash
# One pass per quant type — each lands in its own workspace.
uv run python scripts/reproduce_table.py \
    --model Qwen/Qwen3.5-4B --quant-type Q4_K_M \
    --logs logtrain.jsonl --wiki-test-raw /path/to/wiki.test.raw

uv run python scripts/reproduce_table.py \
    --model Qwen/Qwen3.5-4B --quant-type IQ4_NL \
    --logs logtrain.jsonl --wiki-test-raw /path/to/wiki.test.raw

# Or chain both quant types for Qwen3.5-4B back-to-back:
uv run python scripts/reproduce_table_qwen3_5_4b.py \
    --logs logtrain.jsonl --wiki-test-raw /path/to/wiki.test.raw
```

The reproducer goes through 7 stages: F16 → 2 corpora (custom / wiki) → 4 imatrix variants ({stock, hybrid} × {custom, wiki}) → 5 quants (`<TYPE>-none` + 4 calibrated) → KLD + 10-rep speed bench → tool-call holdout + reps → MMLU-Pro reps → `LEADERBOARD.md`. Pass `--skip-toolcall`, `--skip-mmlu`, or `--skip-speed-rebench` to scope down; `--toolcall-reps N` / `--mmlu-reps N` shrink the long stages.

## Requirements

* Python ≥ 3.11
* `uv` (`brew install uv` or `pipx install uv`)
* A C++ toolchain for building `llama.cpp` (Xcode CLT on macOS, or build-essential on Linux)

## Quick start

```bash
# 1. Clone + fetch the pinned llama.cpp submodule
git clone <this repo> quant-tuner
cd quant-tuner
git submodule update --init --recursive

# 2. Build llama.cpp once. Metal on macOS; swap for -DGGML_CUDA=ON on Linux+NVIDIA.
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON
cmake --build vendor/llama.cpp/build -j

# 3. Install the Python package
uv sync
```

## Pipeline at a glance

```
   HF model         usage logs (.jsonl)
       │                     │
       ▼                     ▼
   extract             ingest + split
       │                     │
       │            train.txt | test.txt | holdout.jsonl
       │                     │
       └──── HF → F16 GGUF ◄─┘
                  │
                  │  calibrate (imatrix | AWQ | GPTQ)
                  ▼
       imatrix.gguf  /  awq.pt  /  hf_model_gptq/
                  │
                  │  llama-quantize  (--type Q4_K_M, IQ4_XS, …)
                  ▼
              model.gguf
                  │
                  │  bench (KLD vs FP16, PPL, BPW, prefill/decode tok/s)
                  ▼
              results.csv  →  LEADERBOARD.md
```

## Running an end-to-end calibration

The CLI is recipe-driven. Each recipe under `src/quant_tuner/recipes/` declares
one calibration method × quant type:

```bash
# Method = imatrix (hybrid_custom variant), quant = Q4_K_M
uv run quant-tuner run \
    --recipe q4_k_m_imatrix \
    --model Tesslate/OmniCoder-9B \
    --logs logtrain.jsonl \
    --workspace ./out/my_run

# Validate-only (resolves the recipe + overrides, prints the merged config):
uv run quant-tuner run --recipe q4_k_m_imatrix --model X --logs Y --workspace W --dry-run
```

Available recipes: `q4_k_m_imatrix`, `q4_k_m_awq`, `q4_k_m_gptq`, `q4_k_m_none`,
`q4_k_m_qwen3_5_4b`, `q4_k_m_qwen3_6_mtp`. A recipe is just YAML — copy any of
them to a local file and pass the path to `--recipe` to override the
calibration variant or sampling params.

### Calibration token budget

Each recipe declares two token budgets under `data:`:

```yaml
data:
  train_max_tokens: 250000        # cap for the calibration corpus
  eval_max_tokens:  50000         # cap for the KLD/PPL eval slice
  supplement: ./calibration_supplement.txt   # optional; appended to train.txt
```

The stratified packer fills `train.txt` up to `train_max_tokens` (a little
over is allowed — sessions can spill past the cap by ≤10%) and `eval.txt` up
to `eval_max_tokens`, both drawn from disjoint splits of the same log file.
If `supplement` is set, the file is appended verbatim to the train corpus
after packing — useful for language coverage (Rust, SQL, shell, etc.) that's
under-represented in your usage logs. The bundled
`calibration_supplement.txt` is the one used for the published results.

### Qwen3.5-4B example

```bash
uv run python scripts/quantize_qwen3_5_4b.py --logs logtrain.jsonl
# → out/q4_k_m_qwen3_5_4b/gguf/Q4_K_M-imatrix.gguf
```

The recipe (`q4_k_m_qwen3_5_4b.yaml`) targets `Qwen/Qwen3.5-4B`, uses the
`hybrid_custom` imatrix variant, and applies the 250K / 50K token budgets
plus the bundled supplement.

## Task-level benchmarks

Two task evals live alongside the KLD/speed bench, both reporting
**mean ± stdev across N reps** for a configurable N (default 10):

```bash
# Tool-calling on a held-out session corpus (built from logtrain.jsonl)
uv run python scripts/run_toolcall_reps.py \
    --models out/run/model-f16.gguf out/run/Q4_K_M-none.gguf \
    --reps 10

# MMLU-Pro CS + math (25 random questions per subject, 2-shot)
uv run python scripts/build_mmlu_pro_holdout.py    # one-time, downloads dataset
uv run python scripts/run_mmlu_pro_reps.py \
    --models out/run/model-f16.gguf out/run/Q4_K_M-none.gguf \
    --reps 10
```

Both runners spawn one `llama-server` per model and reuse it across reps; the
per-rep seed is `--base-seed + rep_idx` for reproducibility. Output: one CSV
per rep + one aggregated CSV (mean / stdev / n per metric). Plug a new
benchmark into this pipeline by writing a `dict[str, float]`-returning adapter
on top of `quant_tuner.eval.reps.run_reps_for_models` — see
`docs/benchmarks.md` § "Multi-rep eval".

## Python API

The pipeline functions are also importable for ad-hoc scripting:

```python
from pathlib import Path

from quant_tuner.calibrate import imatrix, awq, gptq
from quant_tuner.quantize import convert, gguf
from quant_tuner.bench import runner, bpw, kld
from quant_tuner.models import llama_cpp

work = Path("./out")
model_dir = Path("./model")        # local HF checkpoint
corpus = Path("./corpus.train.txt")  # one big text file of calibration tokens
eval_ds = Path("./corpus.test.txt")  # held-out tokens for KLD/PPL

# 1. HF -> F16 GGUF (one-time)
f16 = convert.hf_to_f16_gguf(model_dir, work / "model-f16.gguf")
ref_baseline = kld.build_baseline(f16, eval_ds, work / "baseline.kld")

# 2a. imatrix path
llama_cpp.imatrix(f16, corpus, work / "imatrix-custom.gguf")
imatrix.calibrate(
    variant="hybrid_custom",
    f16_gguf=f16,
    base_imatrix=work / "imatrix-custom.gguf",
    out_path=work / "imatrix-tuned.gguf",
)
gguf.quantize(f16, work / "Q4_K_M-imatrix.gguf", "Q4_K_M",
              imatrix=work / "imatrix-tuned.gguf")

# 2b. AWQ path
awq.calibrate(model_dir, corpus, work / "awq.pt", force_alpha=0.5)
awq.apply(model_dir, work / "awq.pt", work / "model_awq")
f16_awq = convert.hf_to_f16_gguf(work / "model_awq", work / "model-f16-awq.gguf")
gguf.quantize(f16_awq, work / "Q4_K_M-awq.gguf", "Q4_K_M",
              imatrix=work / "imatrix-custom.gguf")

# 2c. GPTQ path
gptq.calibrate(model_dir, corpus, work / "hessians")
gptq.apply(model_dir, work / "hessians", work / "model_gptq")
f16_gptq = convert.hf_to_f16_gguf(work / "model_gptq", work / "model-f16-gptq.gguf")
gptq.verify_perplexity(f16_gptq, eval_ds, reference_ppl=baseline_ppl, max_ratio=1.5)
gguf.quantize(f16_gptq, work / "Q4_K_M-gptq.gguf", "Q4_K_M",
              imatrix=work / "imatrix-custom.gguf")

# 3. Bench every quant
n_params = bpw.n_params(f16)
for label, quant in [
    ("imatrix", work / "Q4_K_M-imatrix.gguf"),
    ("awq",     work / "Q4_K_M-awq.gguf"),
    ("gptq",    work / "Q4_K_M-gptq.gguf"),
]:
    row = runner.bench_one(
        quant, label,
        reference_n_params=n_params,
        eval_dataset=eval_ds,
        eval_baseline=ref_baseline,
        suite="full",
    )
    runner.append_row(work / "results.csv", row)
```

Switching the target quant type is a one-string change — `gguf.quantize(...,
"IQ4_XS")` or `"Q5_K_M"` works identically. See `docs/methods.md` for the
tradeoffs between common K-quant and I-quant tags.

## Layout

```
src/quant_tuner/
├── calibrate/        # imatrix | awq | gptq calibrators
├── quantize/         # HF → F16 GGUF, F16 → Q* GGUF
├── bench/            # bpw | kld | speed | runner (CSV row builder)
├── data/             # log ingest, stratified packing, train/test/holdout split
├── eval/             # task-level evals (toolcall, mmlu_pro) + generic N-rep runner
├── experiments/      # shared log/phase/step helpers for driver scripts
├── leaderboard/      # CSV → markdown aggregation with SQS scoring
├── models/           # HF extract, llama.cpp binary wrappers, HF→GGUF name map
├── recipes/          # YAML recipes consumed by `quant-tuner run --recipe ...`
├── cli.py            # typer CLI: run | bench | leaderboard
└── pipeline.py       # end-to-end pipeline: extract → calibrate → quantize → bench

vendor/llama.cpp      # pinned submodule, commit 45b455e6
tests/unit/           # 100+ passing tests
```

## Status

Beta. End-to-end calibration runs via the CLI (`quant-tuner run --recipe …`) or
the Python API. Tool-call evaluation lives in `quant_tuner.eval`. The OmniCoder
leaderboard reproducer (`scripts/reproduce_leaderboard.py`) chains the eight
study artifacts end-to-end.

## Pinned llama.cpp

This repo vendors `llama.cpp` at commit `45b455e66fc09abed65b7d52d42a4a29ba0d45d6`
as a git submodule under `vendor/llama.cpp`. Override the build location with
`LLAMA_CPP_DIR=/path/to/your/build` if you'd rather use a system install.
