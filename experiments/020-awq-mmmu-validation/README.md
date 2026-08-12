# Experiment 020 — AWQ cv-gate with MMMU validation corpus

## Why redo this

exp-019 nailed the disjointness invariant for cal / val / eval, but the
validation slice itself was small (~10K tokens of logtrain *test* +
`calibration_supplement.txt`). The binary cv-gate inside
`awq.calibrate(cv_strategy="gate", ...)` makes one accept/reject decision
per tensor based on the proxy loss on that val slice. With only ~10K
tokens, the proxy-loss difference between a per-tensor α candidate and
the group α is dominated by sampling noise, so the gate fires inconsistently
and the per-tensor refinement degenerates toward "always accept" or "always
reject" in different layers.

A larger, more genuinely out-of-distribution val slice should:

1. Reduce sampling-noise dominance in the per-tensor / group α delta,
   making the gate more confident.
2. Give the per-tensor α a fairer test — α candidates that win on cal
   purely because they overfit logtrain stylistic patterns should now lose
   on the MMMU val proxy.

## What changes here

Swap **only the validation source**:

- **cal** (unchanged from exp-019) — ALL wiki.test.raw + ~500K logtrain
  *train* tokens.
- **val** — `calibration_supplements/mmmu/combined.txt` (MMMU disciplines,
  ~100–200K tokens). The logtrain *test* slice is dropped entirely so val
  is purely OOD.
- **eval** (unchanged from exp-019) — ~30K tokens each from external
  `eaddario/imatrix-calibration` `{code_small, math_small, tools_small}`.

The AWQ scoring logic in `src/quant_tuner/calibrate/awq.py` is
**unchanged**. Only the path passed as `holdout_text` differs.

Only the `cv-gate` variant runs (the release ships cv-gate). The
imatrix-only baselines and the plain Q2_K anchor are re-quantized and
re-benched fresh in exp-020 so the experiment is self-contained — though
both rows are expected to match exp-019 within bench noise, since neither
depends on validation data.

## Setup

- Model: `google/gemma-4-31B-it` (HF + F16 GGUF reused from exp-009).
- Quants: `IQ2_XS`, `IQ2_M`, `Q2_K_S`; plus a plain `Q2_K` anchor.
- imatrix: re-computed on the new `corpus.cal.txt` (cal contents identical
  to exp-019, so the imatrix should match — re-running guards against the
  case where exp-019 outputs are unavailable).
- Baseline KLD: re-computed on `corpus.eval.txt` (same external eval
  domains, same per-domain target tokens, same seed → byte-identical to
  exp-019).
- `sanity_max_rel` for `awq.apply` is loosened from 0.85 → 0.95: the much
  larger val corpus may shift selected α further from the group value, and
  that is the expected behavior.

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/run_exp020_awq_mmmu_validation.py
```

Idempotent via `experiments.step()`. Outputs go to
`out/exp-020/google__gemma-4-31B-it/`:

- `corpora/{corpus.cal,corpus.val,corpus.eval,corpora_audit.json,...}`
- `imatrix-cal.gguf`, `baseline.kld`
- `gate/{awq.pt,model-f16-awq.gguf,IQ2_XS-awq.gguf,IQ2_M-awq.gguf,Q2_K_S-awq.gguf,results.csv}`
- `imatrix-only/{IQ2_XS-imatrix.gguf,IQ2_M-imatrix.gguf,Q2_K_S-imatrix.gguf,results.csv}`
- `plain/{Q2_K-plain.gguf,results.csv}`
- `table.md` — the §3 comparison table for the upload README.

## After-the-fact

- `scripts/plot_awq_cv_gate_release.py --results-root out/exp-020/google__gemma-4-31B-it --out <upload>/awq_cv_gate_release.png`
- `scripts/plot_awq_cv_gate_release_scatter.py --results-root out/exp-020/google__gemma-4-31B-it --out <upload>/awq_cv_gate_release_scatter.png`
- `scripts/update_release_assets_exp020.py` — patches the §3 table and §2.3
  data-slice description in the upload README; copies the three new GGUFs
  in.
