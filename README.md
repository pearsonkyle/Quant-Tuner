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
