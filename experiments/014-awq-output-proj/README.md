# Experiment 011 — AWQ on output projections (o_proj + down_proj)

## Hypothesis

The naive AWQ port only scales `q/k/v/gate/up` (5 of 7 linear projections per
layer). The two output projections — `o_proj` (attention output) and
`down_proj` (MLP output) — are the largest tensors in the model and are left
untouched. At sub-3 bpw (IQ2_XS, IQ2_M, Q2_K_S) this is the most degraded
weight in the network with no input-side rescaling.

This experiment adds `o_proj` and `down_proj` as their own single-member
scale groups. Their inputs are not normalized activations, so there is no
RMSNorm γ to fold into — scales are applied to the weight only and the F16
forward intentionally drifts (sanity check bounded by `sanity_max_rel=0.25`).

## Setup

- Model: `google/gemma-4-31B-it`
- Quants: `IQ2_XS`, `IQ2_M`, `Q2_K_S`
- Reuses exp-009 HF source, F16 GGUF, mixed8k corpus, imatrix, eval, baseline KLD
- Compares against exp-009 (imatrix only) and exp-010 (naive AWQ)

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/run_exp014_awq_output_proj.py
```

Read results:

```bash
cat out/exp-014/google__gemma-4-31B-it/table.md
```

## Success criteria

IQ2_XS or IQ2_M PPL improves vs exp-010 by ≥3% without KLD regressing by
more than 5%.
