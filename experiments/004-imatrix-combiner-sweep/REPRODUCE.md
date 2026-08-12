# Reproducing experiment 004: imatrix combiner sweep

## Environment

Same as exp-001 / exp-002 / exp-003.

## Prerequisites

Exp-001 must be done (it is). Reuses:

- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/model-f16.gguf`
- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/imatrix-custom.gguf` (vanilla base)
- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/corpus.eval.txt`
- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/baseline.kld`
- `out/exp-002/toolcall_holdout.jsonl`

No HF download, no new llama-imatrix passes, no forward-stats compute.
Every combiner cell only does a CPU-side numpy reweight of the existing
base imatrix.

## Setup

```bash
git checkout -b exp/004-imatrix-combiner-sweep
```

## Steps

1. **Dry-run** to print the planned 5 cells:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp004.py --dry-run
   ```

2. **Smoke** the cheapest cell (anything — all combiners are fast):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp004.py --only C3
   ```
   Builds the imatrix, quantizes, benches. ~3 min.

3. **Full bench sweep** (5 cells, ~15 min total):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp004.py
   ```

4. **Tool-call eval** on the 5 new GGUFs (5 reps × 5 models × ~5 min ≈ 2 h):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_toolcall_reps.py \
     --models \
       out/exp-004/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-C1.gguf \
       out/exp-004/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-C2.gguf \
       out/exp-004/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-C3.gguf \
       out/exp-004/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-C4.gguf \
       out/exp-004/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-C5.gguf \
     --holdout out/exp-002/toolcall_holdout.jsonl \
     --reps 5 --base-seed 1000 \
     --results out/exp-004/toolcall_reps_results.csv \
     --aggregated out/exp-004/toolcall_reps_aggregated.csv \
     --log-dir out/exp-004/logs
   ```

5. **Render the tables** (extends the renderer or just paste from the CSVs):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/render_exp004_tables.py
   ```

## Expected output

```
out/exp-004/Jackrong__Qwopus3.5-9B-Coder/
├── imatrix-C1.gguf … imatrix-C5.gguf
├── Q4_K_M-C1.gguf  … Q4_K_M-C5.gguf
├── results.csv                                # 5 rows, bench
└── logs/*.log
out/exp-004/
├── results.csv                                # aggregated bench
├── toolcall_reps_results.csv                  # per-rep tool-call
├── toolcall_reps_aggregated.csv               # aggregated tool-call
└── TABLES.md
```

## Estimated wall time

- Imatrix build (per cell): < 5 s (CPU numpy)
- Quantize Q4_K_M: ~30 s
- Bench KLD: ~70 s
- Tool-call reps (per model): ~25 min

Total: **~2.5 h** (mostly tool-call eval).
