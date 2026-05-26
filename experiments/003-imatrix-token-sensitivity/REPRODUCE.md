# Reproducing experiment 003: imatrix token & sequence-length sensitivity

## Environment

Same as exp-001 / exp-002: macOS + Metal (or Linux + CUDA), Python 3.13 via
the conda `llm` env, `PYTHONPATH=src .venv/bin/python` (not `uv run`),
llama.cpp built at `vendor/llama.cpp/build`.

Disk: ~12 GB additional (one Q4_K_M GGUF per cell × 11 cells × ~5.24 GiB,
minus overlap; intermediate imatrix `.gguf` files are small). The F16
GGUF and HF cache from exp-001 are reused — no new model downloads.

## Prerequisites

Exp-001 must be done (it is — `status: done` in `research.json`). The
following artifacts under
`out/exp-001/Jackrong__Qwopus3.5-9B-Coder/` are reused by every cell:

- `model-f16.gguf`
- `corpus.eval.txt`  (50K-token holdout-slice eval set)
- `baseline.kld`    (F16 reference)
- `model_extracted/` (HF tokenizer source)

Also reused (top-level):

- `out/exp-001/wiki/wiki.test.raw` (used by A2 and A3)

## Setup

```bash
git checkout -b exp/003-imatrix-token-sensitivity
```

No additional dependencies. `scripts/run_exp003.py` (the driver) is
checked in alongside this doc.

## Steps

1. **Dry-run the full plan** to confirm what will execute:

   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp003.py --dry-run
   ```

   Prints all 11 cells and their target paths without invoking llama.cpp.

2. **Smoke a single cell** before the full sweep:

   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp003.py --only B11
   ```

   B11 (100K tokens, ctx=8K) is the cheapest cell (~5 min). Confirms
   imatrix → quantize → bench wiring end-to-end and lands one row in
   `out/exp-003/results.csv`.

3. **Full sweep** (estimated wall time below; idempotent):

   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_exp003.py
   ```

   Cells with existing outputs are skipped. Order: sub-experiment A first
   (3 cells), then sub-experiment B (9 cells, with B31 deduped against
   A1).

4. **Render the two tables**:

   ```bash
   PYTHONPATH=src .venv/bin/python scripts/render_exp003_tables.py
   ```

   Writes `out/exp-003/TABLES.md`. Paste the two sections into
   `README.md` under the existing Metrics headings.

5. **Status:**

   ```bash
   researcher exp status 3 done
   ```

## Estimated wall time

Per-cell time is dominated by `llama-imatrix` (linear in total tokens) and
the bench KLD pass (~70 s, constant). `ctx` affects per-token throughput
but not total tokens, so total imatrix wall time scales mostly with the
token count.

| Cell | tokens | ctx | imatrix | bench/quant | total |
|------|--------|-----|---------|-------------|-------|
| A1=B31 | 500K | 8K  | ~20 min | ~2 min     | ~22 min |
| A2   | ~280K | 8K  | ~11 min | ~2 min     | ~13 min |
| A3   | ~780K | 8K  | ~31 min | ~2 min     | ~33 min |
| B11  | 100K  | 8K  | ~4 min  | ~2 min     | ~6 min  |
| B12  | 100K  | 16K | ~4 min  | ~2 min     | ~6 min  |
| B13  | 100K  | 20K | ~4 min  | ~2 min     | ~6 min  |
| B21  | 250K  | 8K  | ~10 min | ~2 min     | ~12 min |
| B22  | 250K  | 16K | ~10 min | ~2 min     | ~12 min |
| B23  | 250K  | 20K | ~10 min | ~2 min     | ~12 min |
| B32  | 500K  | 16K | ~20 min | ~2 min     | ~22 min |
| B33  | 500K  | 20K | ~20 min | ~2 min     | ~22 min |

**Total: ~3 hours** for the full 11-cell sweep. `ctx=20K` cells may
fluctuate — if memory pressure forces llama-imatrix into split passes,
add ~20% to those rows.

## Expected output

```
out/exp-003/
├── results.csv                                       # 11 rows, aggregated
├── TABLES.md                                         # rendered for README
└── Jackrong__Qwopus3.5-9B-Coder/
    ├── corpus.custom_100K.txt                        # cached per-token-target
    ├── corpus.custom_250K.txt
    ├── corpus.custom_500K.txt                        # symlink to exp-001 if compatible
    ├── corpus.combined.txt                           # custom_500K + wiki concat (A3)
    ├── imatrix.A1.gguf  (etc., one per cell)
    ├── Q4_K_M.A1.gguf   (etc.)
    ├── results.csv
    └── logs/*.log
```

Per-cell bench line on success:

```
  size=5.24 GiB bpw=5.029 ppl=... mean_kld=... same_top_p=...
```

## Caveats

- **Wiki file is fixed at ~280K tokens.** A2 and A3 do not get a 500K-token
  wiki version — the comparison in sub-experiment A is intentionally not
  token-matched (see README "Approach" section).
- **`per_session_cap=6000` is held fixed** across all custom-corpus cells.
  Increasing it would let `ctx=20K` cells pack more per session, which
  would confound the ctx axis.
- **`stratified_pack` seed is fixed at 42** for `target_tokens` ≥ 100K so
  smaller-token cells are deterministic subsets, not different random
  draws. (Different `target_tokens` values share session selection up to
  the cap.)
- **The eval set is identical to exp-001's KLD eval** (the `holdout` slice
  of `logtrain.jsonl`). It was NOT used for calibration in any of these
  cells, but it IS the same set exp-001 and exp-002 used — so KLD numbers
  from this experiment are directly comparable to those tables.
