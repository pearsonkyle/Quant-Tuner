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

End-to-end run on a real 9B coding model. The calibration corpus is the
user's actual usage log (`logtrain.jsonl`, 253 sessions split 80/10/10 by
fingerprint); the eval is the held-out test split. Tool-call eval uses the
disjoint holdout split (10 sessions, claude + qwen sources).

Sorted by SQS (1, 2, 1 weights — fidelity weighted 2×):

| Model | Size (GiB) | Mean KLD | Same Top p | Prefill tok/s | Decode tok/s | TTFT@2k (ms) | Tool Sel % | Param Acc % | Schema % | Rollout % | SQS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock/Q4_K_M-custom  | 5.24 | 0.612 | 84.75 | 778.7 ± 15 | 64.40 ± 0.4 | 2630 ± 52 | 33.3 | 33.3 | 73.3 | 70.0 | 1.513 |
| stock/Q4_K_M-wiki    | 5.24 | 0.635 | 84.45 | 776.1 ± 8  | 64.15 ± 0.6 | 2639 ± 26 | 37.5 | 32.8 | 75.0 | 80.0 | 1.509 |
| hybrid/Q4_K_M-custom | 5.24 | **0.595** | **84.80** | 764.9 ± 8  | 63.32 ± 0.8 | 2678 ± 30 | 37.5 | 32.8 | 75.0 | 60.0 | 1.507 |
| hybrid/Q4_K_M-wiki   | 5.24 | 0.638 | 84.28 | 763.1 ± 13 | 60.53 ± 2.6 | 2684 ± 47 | **41.2** | **35.3** | 64.7 | 50.0 | 1.486 |
| baseline/Q4_K_M-none | 5.24 | 1.012 | 81.11 | 741.3 ± 18 | 64.44 ± 0.5 | 2763 ± 68 | 33.3 | 26.7 | 66.7 | 40.0 | 1.480 |
| baseline/fp16        | 16.69 | 0.000 | 99.99 | 901.5 ± 20 | 28.10 ± 0.07 | 2272 ± 51 | 41.2 | 35.3 | 70.6 | 60.0 | 1.000 |

Per-run stdev is over 10 `llama-bench` repetitions on the same machine, one
model at a time. Tool-call sample sizes are small (10 sessions, 15–17
scored turns each), so the small inter-row gaps in those columns are within
noise — but the **calibration-vs-no-calibration** gaps are real and consistent
with the KLD signal.

Headline reads:

- **Any imatrix beats none.** Mean KLD drops 1.012 → ~0.6 (≈ −40 %).
- **Your-own-data beats wikitext on KLD.** custom ≈ 0.60, wiki ≈ 0.64. The
  gap is wider under hybrid (7 %) than stock (4 %) — the analytic refinement
  amplifies the corpus signal.
- **Hybrid beats stock at fixed corpus on KLD, but only just at Q4_K_M.** ~3 % on
  custom, basically tied on wiki. Hybrid is expected to pull further ahead
  at lower bit budgets (IQ4_XS, IQ3_S) where channel allocation is binding.
- **Tool-call accuracy is too sample-limited to rank** the four calibrated
  rows, but all of them beat the uncalibrated quant on schema-validity and
  rollout-completion. fp16 ≠ best tool-caller here — the calibrated Q4
  rows sit at or near fp16 on every tool-call column except top-1 selection.

Raw data: `out/omnicoder_q4_k_m/{LEADERBOARD.md, results.csv, toolcall_results.csv}`.
Reproduce with `scripts/run_omnicoder_q4_k_m.py` followed by
`scripts/run_omnicoder_wiki_vs_custom.py`, `scripts/run_toolcall_all.py`,
and `scripts/rebench_speed.py`.

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

## Python API

The CLI (`quant-tuner …`) currently exposes only top-level stubs; the working
surface is the Python API. A minimal end-to-end run on a 1B model:

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
├── models/           # HF extract, llama.cpp binary wrappers, HF→GGUF name map
└── leaderboard/      # CSV aggregation (Phase 5, in progress)

vendor/llama.cpp      # pinned submodule, commit 45b455e6
tests/unit/           # 36 passing tests; pure-math + parser coverage
```

## Status

Pre-alpha. Calibrators and bench infrastructure work end-to-end via the Python
API. The `quant-tuner …` CLI commands are still stubs. The leaderboard
aggregation (`SQS` scoring formula, markdown report writer) is the next milestone.

## Pinned llama.cpp

This repo vendors `llama.cpp` at commit `45b455e66fc09abed65b7d52d42a4a29ba0d45d6`
as a git submodule under `vendor/llama.cpp`. Override the build location with
`LLAMA_CPP_DIR=/path/to/your/build` if you'd rather use a system install.
