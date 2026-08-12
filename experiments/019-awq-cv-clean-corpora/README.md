# Experiment 019 — AWQ cv-mixed / cv-gate on disjoint corpora

## Why redo this

exp-017 (cv-gate) and exp-018 (cv-mixed) tested whether a held-out signal
could regularize per-tensor α selection. Both used:

- **Calibration:** `corpus.mixed8k.txt` — logtrain *train* + wiki.test.raw.
- **Held-out:** `out/holdout_chunks/cv_1k.txt` — 1K tokens from logtrain *test*.
- **Eval (bench):** `corpus.eval.txt` — also derived from logtrain + wiki.

Cal and held-out are two random draws of the same logtrain distribution.
A score like

```
score(α) = L(W, X_cal, α) + cv_weight · L(W, X_ho, α)
```

with correlated `X_cal` and `X_ho` is just `(1 + cv_weight)` times a noisy
single-distribution loss; the held-out term doesn't measure generalization,
and `cv_weight=2.0` (exp-018) is mathematically arbitrary in that regime.
Worse, the bench `corpus.eval.txt` overlaps the calibration distribution,
so PPL/KLD conflate fit with generalization.

## What changes here

Wire the existing cv-mixed and cv-gate AWQ pipelines into the disjoint
corpora produced by `scripts/build_corpora.py`:

- **cal** = ALL wiki.test.raw + ~500K logtrain *train* tokens.
- **val** (held-out) = ~10K logtrain *test* tokens + `calibration_supplement.txt`
  (under-represented content — Rust, JSON, YAML — so val is a real
  distribution shift, not a re-draw).
- **eval** (bench) = ~30K tokens each from external
  `eaddario/imatrix-calibration` `{code_small, math_small, tools_small}`.
  Neither logtrain nor wiki appears in eval, so PPL/KLD measure
  generalization rather than recall of calibration data.

Scoring logic (`cv_strategy="mixed"` with `cv_weight=2.0`; `cv_strategy="gate"`)
is **unchanged** from exp-017 / exp-018. The point of exp-019 is to test
whether that scoring carries real signal when cal and val are actually
disjoint distributions.

## Setup

- Model: `google/gemma-4-31B-it` (HF + F16 GGUF reused from exp-009).
- Quants: `IQ2_XS`, `IQ2_M`, `Q2_K_S`.
- imatrix: re-computed on the new `corpus.cal.txt` (old imatrix was on the
  old cal corpus).
- Baseline KLD: re-computed on the new `corpus.eval.txt` (old baseline was
  against the old eval corpus).
- AWQ apply uses `sanity_max_rel=0.85` (matches exp-018; revisit if it trips).

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/run_exp019_awq_cv_clean.py
```

Idempotent via `experiments.step()` — re-running skips already-produced
artifacts.

Read results:

```bash
cat out/exp-019/google__gemma-4-31B-it/table.md
```

## Success criteria

- **`corpora_audit.json` shows disjoint logtrain splits** and the target
  token counts (~500K cal, ~10K val, ~90K eval).
- **PPL in a sane range** for IQ2_M (roughly 300–2500 on the new code/math/tools
  eval; a collapse > 5000 means α selection destabilized under the new val
  distribution — a finding either way).
- **KLD ≤ exp-010 naive-AWQ baseline on the new eval corpus.**
- **cv-mixed and cv-gate produce different α tables** than exp-017 / 018 on
  the same model — confirmable from `awq.pt` metadata. If they don't, the
  val data isn't moving the optimizer; the technique is inert.

If exp-019 cv-mixed/cv-gate fail to improve on naive AWQ even with clean
disjoint corpora, the conclusion is that the held-out-loss technique
itself doesn't carry signal for sub-3-bpw Gemma — independent of data
hygiene. Follow-up would be a `cv_weight` sweep (exp-020), not another
data shuffle.
