# Experiment 013 — Per-tensor α refinement

## Hypothesis

q/k/v share an input channel statistic so they must share an activation
profile, but their weight statistics differ. The naive AWQ pass locks one
α per group. After the group α is chosen, this experiment runs a second
pass: each member independently picks the α (from a local grid
`group_α ± 0.25`, clipped to `[0, 1]`) that minimizes its own proxy
reconstruction loss.

The group scale is still what gets folded into the RMSNorm gain (one γ, can
only cancel one scale). Each member's weight is multiplied by its own
per-member scale; the residual ratio is a controlled F16 perturbation
(sanity-bounded by `sanity_max_rel=0.20`).

## Setup

- Model: `google/gemma-4-31B-it`
- Quants: `IQ2_XS`, `IQ2_M`, `Q2_K_S`
- Reuses exp-009 artifacts; compares against exp-009 and exp-010

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/run_exp016_awq_per_tensor_alpha.py
```

Read results:

```bash
cat out/exp-016/google__gemma-4-31B-it/table.md
```

## Success criteria

IQ2_XS or IQ2_M PPL drops vs exp-010 by ≥3% without KLD regression. Bonus:
inspect calibration log for per-member α histogram by tensor type — if all
members converge to the group α, refinement adds no signal.
