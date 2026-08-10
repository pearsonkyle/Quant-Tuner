# Reproducing experiment 006: outlier_max rescue + heavy-tail signal mixing

## Environment

Same as exp-001 / exp-002 / exp-003 / exp-004 / exp-005.

## Prerequisites

Exp-001 done (vanilla custom imatrix + F16 GGUF + KLD baseline).
Exp-002 outlier extension done with the **full mapping fix** (so that
`forward_stats.npz` reflects 100% non-SSM coverage). Required artifacts:

- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/model-f16.gguf`
- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/imatrix-custom.gguf`  (vanilla base)
- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/corpus.eval.txt`
- `out/exp-001/Jackrong__Qwopus3.5-9B-Coder/baseline.kld`
- `out/exp-002/Jackrong__Qwopus3.5-9B-Coder/forward_stats.npz` (full-mapping E[a⁴] + max|a|)
- `out/exp-002/toolcall_holdout.jsonl`

No HF download, no new llama-imatrix passes, no new forward pass.
Every cell is numpy reranking + quantize + bench.

## Setup

```bash
git checkout -b exp/006-heavy-tail-mixing
```

## Steps

1. **Run the diagnostic first** (~5 s, no GGUF builds):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/diagnose_outlier_max.py
   ```
   Writes `out/exp-006/diagnostic.md` and prints to stdout. Use the
   output to decide whether the planned Phase B cells make sense, or
   whether the tensor-class fallback boundaries need adjusting.

2. **Dry-run the planned cells**:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp006.py --dry-run
   ```

3. **Smoke one cell** (~3 min):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp006.py --only F1
   ```

4. **Full bench sweep** (7 cells, ~25 min):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp006.py
   ```

5. **Tool-call eval** on the 7 new GGUFs (5 reps × 7 models ≈ 3 h):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_toolcall_reps.py \
     --models \
       out/exp-006/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-F1.gguf \
       out/exp-006/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-F2.gguf \
       out/exp-006/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-F3.gguf \
       out/exp-006/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-H1.gguf \
       out/exp-006/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-H2.gguf \
       out/exp-006/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-H3.gguf \
       out/exp-006/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-H4.gguf \
     --holdout out/exp-002/toolcall_holdout.jsonl \
     --reps 5 --base-seed 1000 \
     --results out/exp-006/toolcall_reps_results.csv \
     --aggregated out/exp-006/toolcall_reps_aggregated.csv \
     --log-dir out/exp-006/logs
   ```

6. **Recompute pass@5** (picks up the new JSONL files automatically):
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/compute_toolcall_passat5.py
   ```
   _Note_: `RUN_SPECS` in that script will need a new entry for
   `out/exp-006/toolcall_reps_results.csv` so its files get accounted
   for in the chronological mapping.

7. **Status:** `researcher exp status 6 done`.

## Cell-skipping strategy (per Phase A output)

Phase A's diagnostic might reveal that the suspected culprit (extreme
max|a| spikes on attn_qkv / attn_gate) is in fact widespread or
absent. If absent, the F1/F2/F3 cells become uninformative and can
be skipped via `--only H1 H2 H3 H4`. If widespread, consider extending
F3's class scope to more tensor families. **Don't run all 7 cells
blindly** — use the diagnostic to focus.

## Estimated wall time

Per-cell time:
- Imatrix rerank: < 2 s (numpy)
- Quantize Q4_K_M: ~30 s
- Bench KLD: ~70 s
- Tool-call reps per model: ~25 min

Total: **~3.5 h** if running all 7 cells (~25 min bench + ~3 h tool-call).

## Expected output

```
out/exp-006/
├── diagnostic.md                                       # Phase A output
├── results.csv                                          # 7 bench rows
└── Jackrong__Qwopus3.5-9B-Coder/
    ├── imatrix-F1.gguf … imatrix-H4.gguf
    ├── Q4_K_M-F1.gguf  … Q4_K_M-H4.gguf
    ├── results.csv
    └── logs/*.log
```
