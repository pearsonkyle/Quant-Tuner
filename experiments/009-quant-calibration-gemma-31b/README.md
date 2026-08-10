# Experiment 009: Quant calibration sweep on google/gemma-4-31B-it

- **Status:** queued
- **Created:** 2026-06-02
- **Branch:** `exp/009-quant-calibration-gemma-31b` _(create with `git checkout -b exp/009-quant-calibration-gemma-31b`)_

## Summary

Single-model, single-corpus sweep on `google/gemma-4-31B-it`. The
mixed8k corpus (500k custom tokens from `logtrain.jsonl` +
`calibration_supplement.txt` followed by the full `wiki.test.raw`) is
built once, fed to `llama-imatrix` at **ctx=8192**, and reused across
three quant targets. F16 acts as the KLD reference. All KLD/PPL numbers
are produced at **ctx=4096** (gemma's ~262k vocab busts
`llama-perplexity` at ctx=8192).

Quant targets:

- `Q5_K_S`
- `Q3_K_S`
- `IQ2_XXS`

Recommended sampling for evals (per the model card, not used for
calibration): `temperature=1.0`, `top_p=0.95`, `top_k=64`.

## Prerequisites

- `wiki.test.raw` staged under `out/exp-001/wiki/` (or `$WIKI_TEST_RAW`
  set). Reused from exp-001.
- `logtrain.jsonl` + `calibration_supplement.txt` at repo root.
- ≥ ~70 GiB free disk for the F16 GGUF + quants, plus enough VRAM/RAM
  for `llama-imatrix` over a 31B at ctx=8192.

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/run_exp009_quant_calibration_gemma_31b.py
```

Idempotent — every stage is wrapped in `experiments.step()`. Outputs land
under `out/exp-009/google__gemma-4-31B-it/`.

## Quantization Test with calibration data using google/gemma-4-31B-it

_Filled in by `scripts/run_exp009_quant_calibration_gemma_31b.py` (table
written to `out/exp-009/google__gemma-4-31B-it/table.md`)._

| quant | technique | dataset | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |
|---|---|---|---|---|---|---|---|
| FP16    | none    | —             | — | — | — | 0.00000 | 100.0000 |
| Q5_K_S  | imatrix | 500k-custom+wiki (ctx=8192) | — | — | — | — | — |
| Q3_K_S  | imatrix | 500k-custom+wiki (ctx=8192) | — | — | — | — | — |
| IQ2_XXS | imatrix | 500k-custom+wiki (ctx=8192) | — | — | — | — | — |
