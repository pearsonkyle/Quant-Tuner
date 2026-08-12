# Reproducing experiment 001: imatrix with custom data

## Environment

- macOS (Apple Silicon, Metal). For Linux+CUDA, build llama.cpp with `-DGGML_CUDA=ON`.
- Python 3.13 in the conda `llm` env. On this machine, `uv run` resolves the
  wrong interpreter — use `PYTHONPATH=src .venv/bin/python` instead.
- llama.cpp built at `vendor/llama.cpp/build` (submodule pinned to 45b455e6).
- ~80 GB free disk (3 × F16 GGUF + 9 × Q4_K_M GGUF + HF caches).
- `~/.cache/huggingface` writable; HF credentials set if any model gates downloads.
- A local copy of `wiki.test.raw` from
  `https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip`.
  Point `$WIKI_TEST_RAW` at it (or place it at `out/exp-001/wiki/wiki.test.raw`).

## Setup

```bash
git submodule update --init --recursive
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON
cmake --build vendor/llama.cpp/build -j
uv sync
git checkout -b exp/001-imatrix-with-custom-data
export WIKI_TEST_RAW=/path/to/wiki.test.raw    # one-time
```

## Steps

1. **Dry-run for one model / one cell** to confirm wiring:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp001.py \
     --models Qwen/Qwen3.5-9B --only none --dry-run
   ```
   Should print the planned step list and exit 0 without launching llama.cpp.

2. **Full run** (3 models × 3 cells = 9 quants + benches):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp001.py
   ```
   Reruns are safe: every stage is wrapped in `experiments.step()` and skips
   on existing output. Expect hours of wall time on first run (HF download +
   F16 conversion + imatrix builds + benches), seconds on rerun.

3. **Subsetting** while iterating:
   ```bash
   # one model, all three cells
   PYTHONPATH=src .venv/bin/python scripts/run_exp001.py \
     --models Tesslate/OmniCoder-9B
   # one cell across all models
   PYTHONPATH=src .venv/bin/python scripts/run_exp001.py --only custom
   ```

4. **Render the table:**
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/render_exp001_table.py
   ```
   Writes `out/exp-001/TABLE.md` and prints it. Paste under the Metrics
   heading of `README.md`.

5. **Record status:**
   ```bash
   researcher exp status 1 running    # when kicked off
   researcher exp status 1 done       # when the table is filled in
   ```

## Expected output

Per-model layout:

```
out/exp-001/
├── results.csv                              # aggregated, 9 rows
├── TABLE.md                                 # rendered markdown table
├── wiki/wiki.test.raw                       # cached copy (or symlink)
└── {model-slug}/
    ├── model_extracted/                     # text-only HF dir
    ├── model-f16.gguf
    ├── corpus.train.txt                     # logs + supplement
    ├── corpus.eval.txt                      # holdout slice
    ├── baseline.kld                         # F16 reference for KLD
    ├── imatrix-custom.gguf
    ├── imatrix-wiki.gguf
    ├── Q4_K_M-custom.gguf
    ├── Q4_K_M-wiki.gguf
    ├── Q4_K_M-none.gguf
    ├── results.csv                          # per-model, 3 rows
    └── logs/*.log
```

A successful bench row prints, per cell:

```
  size=4.93 GiB bpw=4.78 ppl=… mean_kld=… same_top_p=…
```
