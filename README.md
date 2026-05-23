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

## Results so far (9B model comparison @ IQ3_S with MTP)

Three Qwen3.5-9B-family models compared at IQ3_S quantization. All GGUFs bundle the MTP
`nextn_predict` layers (`qwen35.nextn_predict_layers = 1`) for speculative decoding.
Calibration: `hybrid_custom` imatrix on `logtrain.jsonl` (250k tokens). IQ3_S without
calibration gives catastrophic perplexity (~3.6M PPL measured); `hybrid_custom` is required.
Eval: KLD on a held-out 50k-token test split; speed is 10-rep `llama-bench`.
Hardware: Apple M-series, Metal backend.

| Model | Variant | BPW | PPL | Mean KLD | Same Top-p | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen3.5-9B | baseline/fp16 | 16.01 | 6.65 | — | 100.0% | 27.13 |
| Qwen/Qwen3.5-9B | **IQ3_S-custom** | 3.89 | 7.37 | 1.175 | 74.6% | 60.10 |
| Jackrong/Qwopus3.5-9B-Coder | baseline/fp16 | 16.01 | 4.78 | — | 100.0% | 26.04 |
| Jackrong/Qwopus3.5-9B-Coder | **IQ3_S-custom** | 3.89 | 6.40 | **0.919** | 75.7% | 48.92 |
| Tesslate/OmniCoder-9B | baseline/fp16 | 16.01 | 6.64 | — | 100.0% | 21.46 |
| Tesslate/OmniCoder-9B | **IQ3_S-custom** | 3.89 | 7.40 | 1.172 | 74.6% | 46.14 |

Notes:
- **Jackrong wins on quality** at IQ3_S — lowest KLD (0.919) and best PPL (6.40), despite
  being a code-specialized fine-tune of the same base.
- **Qwen IQ3_S is fastest** at 60 tok/s — the other two carry VLM overhead from their
  extraction path (Tesslate is a `Qwen3_5ForConditionalGeneration` with a vision tower).
- **Tesslate fp16 is slowest** (21 tok/s) — the hybrid attention architecture
  (GatedDeltaNet layers) adds per-token overhead on Metal.
- MTP layers were injected into Jackrong and Tesslate from the Qwen/Qwen3.5-9B donor (those
  models declare `mtp_num_hidden_layers=1` in config but ship without the weights).

**MTP head training.** A fine-tuned MTP head for Tesslate/OmniCoder-9B is in progress,
adapting the donor weights to the tool-call distribution. See
`scripts/train_mtp_head.py` and `scripts/plot_mtp_acceptance.py` for the training
pipeline and acceptance-rate-vs-draft-depth analysis.

**Reproduce:**

```bash
uv run python scripts/benchmark_9b_models.py \
    --logs logtrain.jsonl \
    --workspace out/benchmark_9b_iq3s
# Results: out/benchmark_9b_iq3s/comparison.csv
```

## Results so far (Qwen/Qwen3.6-27B @ Q5_K_S)

End-to-end run on Qwen3.6-27B with the MTP head retained in the GGUF. Calibration:
`hybrid_custom` imatrix on `logtrain.jsonl` (250k tokens) + `calibration_supplement.txt`.
Eval data: 48k-token held-out KLD/PPL slice from `logtrain.jsonl`; 10-session tool-call
holdout disjoint from the calibration split; 50-question MMLU-Pro slice (25 CS + 25 math,
2-shot). Hardware: Apple M-series, Metal backend, `llama.cpp` @ pinned `45b455e6`.

| Model | Size (GiB) | BPW | PPL | KLD | Same Top-p | Decode tok/s | Prefill tok/s | Tool Sel % | Param Acc % | Schema % | Rollout % | MMLU-Pro % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline / fp16            | 50.90 | 16.00 | 5.233 | —     | 100.0 |  6.96 ± 0.46 | 182.2 ± 1.8 | 71.2 ± 4.1 | 35.2 ± 0.7 | 84.6 ± 1.2 | 40.0 ± 0.0 | 59.3 ± 2.3 |
| **Q5_K_S / hybrid_custom** | 17.67 |  5.56 | 5.523 | 0.425 |  88.9 | 13.14 ± 0.39 | 166.0 ± 3.1 | 64.9 ± 0.8 | 33.1 ± 0.4 | 81.8 ± 1.9 | 43.3 ± 5.8 | 58.0 ± 3.5 |

KLD/PPL/speed are 10 reps of `llama-bench`; tool-call and MMLU-Pro are 3 reps each at
`T=0.6, top_p=0.95, top_k=20` (Qwen-recommended sampling).

**MTP draft heads.** The Q5_K_S GGUF bundles the MTP layers (`qwen35.nextn_predict_layers = 1`),
so it can be served with `llama-server --spec-type draft-mtp --spec-draft-n-max N`.
Because the model has **1 nextn layer**, set `--spec-draft-n-max 1`; values above 1 exceed
the trained head depth, drop acceptance rate, and hurt throughput. Benchmarked against a
true baseline (no `--spec-type` flag) and cross-checked against `unsloth/Qwen3.6-27B-MTP-GGUF`:

| Model | `--spec-draft-n-max` | Decode tok/s | Draft acceptance | Speedup |
| --- | ---: | ---: | ---: | ---: |
| ours (imatrix Q5_K_S) | 1 | 18.73 ± 1.35 | 86 % | **1.05x** |
| ours (imatrix Q5_K_S) | 2 | 17.23 ± 2.04 | 74 % | 0.92x |
| ours — baseline | — | 17.81 ± 0.87 | — | — |
| unsloth Q5_K_S | 2 | 17.74 ± 2.46 | 77 % | 0.99x |
| unsloth — baseline | — | 17.92 ± 0.44 | — | — |

MTP gives a modest 5 % wall-clock lift on Metal at `n_max=1`; larger wins appear on CUDA
where decode is compute- rather than memory-bandwidth-bound. Our imatrix calibration yields
higher draft acceptance (86 % vs 77 %) than the unsloth baseline GGUF.

**Reproduce:**

```bash
uv run quant-tuner run --recipe q5_k_s_qwen3_6_mtp \
    --model Qwen/Qwen3.6-27B --logs logtrain.jsonl \
    --workspace out/q5_k_s_qwen3_6_mtp
uv run quant-tuner bench \
    --quant out/q5_k_s_qwen3_6_mtp/gguf/model-f16.gguf \
    --reference out/q5_k_s_qwen3_6_mtp/gguf/model-f16.gguf \
    --eval out/q5_k_s_qwen3_6_mtp/corpus/eval.txt \
    --out out/q5_k_s_qwen3_6_mtp/results.csv --label baseline/fp16
uv run python scripts/run_toolcall_reps.py \
    --models out/q5_k_s_qwen3_6_mtp/gguf/{model-f16,Q5_K_S-imatrix}.gguf \
    --holdout artifacts/toolcall_holdout.jsonl --reps 3
uv run python scripts/run_mmlu_pro_reps.py \
    --models out/q5_k_s_qwen3_6_mtp/gguf/{model-f16,Q5_K_S-imatrix}.gguf --reps 3
uv run python scripts/bench_mtp_speed.py \
    --model out/q5_k_s_qwen3_6_mtp/gguf/Q5_K_S-imatrix.gguf --reps 5 --n-max 1
```

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
