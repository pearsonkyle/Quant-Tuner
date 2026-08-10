# Reproducing experiment 005: block-aware imatrix aggregation

## Environment

Same as exp-001 / exp-002 / exp-003 / exp-004.

## Prerequisites

Exp-001 done (vanilla custom imatrix). Exp-002 partially done — the
`output_aware` imatrix at `out/exp-002/Jackrong__Qwopus3.5-9B-Coder/imatrix-output_aware.gguf`
is needed for cells E3 and E4.

Reused:

- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/model-f16.gguf`
- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/imatrix-custom.gguf`
- `out/exp-002/Jackrong__Qwopus3.5-9B-Coder/imatrix-output_aware.gguf`
- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/corpus.eval.txt`
- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/baseline.kld`
- `out/exp-002/toolcall_holdout.jsonl`

No new corpus prep, no new forward stats, no new llama-imatrix passes.
Block aggregation is pure numpy; rewriting the imatrix GGUF is fast.

## Setup

```bash
git checkout -b exp/005-imatrix-block-aware
```

## Steps

1. **Dry-run** the planned cells:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp005.py --dry-run
   ```

2. **Smoke** one cell:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp005.py --only E2
   ```
   Builds aggregated imatrix → quantize → bench. ~3 min.

3. **Full bench sweep** (5 cells, ~15 min):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp005.py
   ```

4. **Tool-call eval** on the 5 new GGUFs (~2 h):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_toolcall_reps.py \
     --models \
       out/exp-005/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-E1.gguf \
       out/exp-005/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-E2.gguf \
       out/exp-005/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-E3.gguf \
       out/exp-005/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-E4.gguf \
       out/exp-005/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-E5.gguf \
     --holdout out/exp-002/toolcall_holdout.jsonl \
     --reps 5 --base-seed 1000 \
     --results out/exp-005/toolcall_reps_results.csv \
     --aggregated out/exp-005/toolcall_reps_aggregated.csv \
     --log-dir out/exp-005/logs
   ```

## Expected output

```
out/exp-005/Jackrong__Qwopus3.5-9B-Coder/
├── imatrix-E1.gguf … imatrix-E5.gguf
├── Q4_K_M-E1.gguf  … Q4_K_M-E5.gguf
├── results.csv
└── logs/*.log
out/exp-005/
├── results.csv
├── toolcall_reps_results.csv
├── toolcall_reps_aggregated.csv
└── TABLES.md
```

## Estimated wall time

Same shape as exp-004:

- Imatrix aggregation per cell: < 2 s (numpy reshape + max/mean + repeat)
- Quantize: ~30 s
- Bench KLD: ~70 s
- Tool-call reps per model: ~25 min

Total: **~2.5 h** (tool-call eval dominates).

## Sanity check before drawing conclusions

If all E cells produce identical bench numbers to their bases, llama-quantize
is probably aggregating internally already and this experiment is a no-op.
Verify by inspecting one quantized GGUF's per-tensor scales against the
base before reading the tool-call deltas as signal. See "Open question" in
README.md for the relevant llama.cpp source pointers.
