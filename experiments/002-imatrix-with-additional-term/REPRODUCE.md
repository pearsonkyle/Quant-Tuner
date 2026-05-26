# Reproducing experiment 002: imatrix with `output_aware` re-weighting

## Environment

Same as exp-001: macOS + Metal (or Linux + CUDA), Python 3.13 via the
conda `llm` env, `PYTHONPATH=src .venv/bin/python` (avoid `uv run` —
wrong interpreter), llama.cpp built at `vendor/llama.cpp/build`.

## Prerequisite

Exp-001 must have produced these artifacts (which it has — `status: done`):

```
out/exp-001/Jackrong__Qwopus3.5-9B-Coder/
├── model-f16.gguf
├── imatrix-custom.gguf      # vanilla base — input to output_aware
├── corpus.eval.txt          # 50k-token holdout
└── baseline.kld             # F16 reference for KLD
```

No HF download, no new llama-imatrix pass.

## Setup

```bash
git checkout -b exp/002-imatrix-with-additional-term-to-better-behavior
```

## Steps

1. Run the driver (3-minute job):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp002.py
   ```

2. Inspect output:
   ```bash
   cat out/exp-002/results.csv
   ```

3. Update the metrics row in `README.md` with the new numbers, then:
   ```bash
   researcher exp status 2 done
   ```

## Expected output

```
out/exp-002/Jackrong__Qwopus3.5-9B-Coder/
├── imatrix-output_aware.gguf
├── Q4_K_M-output_aware.gguf
├── results.csv                       # one new row
└── logs/*.log
out/exp-002/results.csv               # aggregated
```

Bench line:

```
  size=5.24 GiB bpw=5.029 ppl=… mean_kld=… same_top_p=…
```
