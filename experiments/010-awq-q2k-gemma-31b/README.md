# exp-010: AWQ + sub-3bpw quants on Gemma-4-31B-it

## Question

Does AWQ calibration salvage Q2_K (and other sub-3bpw quants) on
`google/gemma-4-31B-it`? exp-009 ran an imatrix-only sweep that
landed Q2_K at PPL=731 (vs FP16 PPL=302, a 2.4× drift) and saw IQ2_M
collapse to PPL=3917. AWQ's per-channel scaling specifically targets
outlier-heavy channels — the exact regime where 2-bit rounding hurts
most — so it's the most plausible rescue.

## What it does

`scripts/run_exp010_awq_gemma_31b_q2k.py` calibrates AWQ scales on the
mixed8k corpus, folds them into the HF model, converts to F16 GGUF,
then re-quantizes with `llama-quantize` (using **exp-009's existing
imatrix** so the only changed variable is the AWQ fold) for:

- `Q2_K`
- `Q2_K_S`
- `IQ2_M`
- `IQ2_XS`

Bench is KLD + PPL against exp-009's FP16 KLD baseline, so rows are
directly comparable to that experiment's `results.csv`. Output table
puts each quant's imatrix-only row next to its AWQ row.

## Reuses (read-only) from exp-009

- `out/exp-009/.../model_extracted/`
- `out/exp-009/.../model-f16.gguf`
- `out/exp-009/.../corpus.mixed8k.txt`
- `out/exp-009/.../imatrix-mixed8k.gguf`
- `out/exp-009/.../corpus.eval.txt`
- `out/exp-009/.../baseline.kld`

Run exp-009 first; exp-010 fails fast with a clear error if any of
these are missing.

## Gemma-specific config notes

- `rmsnorm_plus_one=False` — Gemma uses Llama-style `γ·x`, not the
  Qwen3.5 `(1+γ)·x` that `recipes/q4_k_m_awq.yaml` defaults to.
- `device="cpu"` — 31B bf16 + Metal overhead won't fit on a 64 GB Mac.
- `eval_ctx=4096` — Gemma's ~262k vocab busts `llama-perplexity` at
  ctx=8192 (same constraint as exp-009).

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/run_exp010_awq_gemma_31b_q2k.py
```

Expect 6–12 h on an M-series Mac. Every stage is idempotent — re-run
to resume.

## Read results

```bash
cat out/exp-010/google__gemma-4-31B-it/table.md
```

Success criteria:
- **Q2_K AWQ PPL < 731** (any improvement validates the approach)
- **IQ2_M AWQ PPL < 3917** (imatrix-only IQ2_M was catastrophic)
- AWQ apply sanity drift < 0.03 (printed by `awq.apply` to stderr)

If Q2_K AWQ looks promising on KLD/PPL, follow up with
`scripts/run_toolcall_reps.py` for the decision-grade signal.
