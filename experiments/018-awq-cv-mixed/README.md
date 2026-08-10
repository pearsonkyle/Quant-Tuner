# Experiment 018 — Per-tensor α with mixed loss (cal + 2·held-out)

> **⚠️ Superseded by exp-019. Do not use these numbers for comparison.**
>
> The cal and held-out corpora were two re-draws of the same logtrain
> distribution, so `score(α) = L_cal + cv_weight · L_ho` was
> `(1 + cv_weight) ×` a noisy single-distribution loss — the held-out
> term carried no generalization signal, and `cv_weight=2.0` was
> mathematically arbitrary in that regime. The bench eval corpus also
> overlapped calibration. exp-019 reruns this technique with disjoint
> cal/val/eval corpora (val now includes `calibration_supplement.txt`,
> eval is external code/math/tools). The big GGUF artifacts (`*.gguf`,
> `model_awq/`) have been deleted to free disk; `awq.pt`, `results.csv`,
> `table.md`, and `logs/` are retained for reference.

## Hypothesis

exp-016 (per-tensor α) over-fit the calibration corpus. exp-017 (binary
held-out gate) is conservative — if the gate rejects most refinements,
per-tensor α has no effect. This experiment instead **scores each α
candidate with a mixed loss**:

```
score(α) = proxy_loss(W, X_cal, s_α) + cv_weight · proxy_loss(W, X_ho, s_α)
```

with `cv_weight=2.0`, i.e. held-out signal is **double-weighted** against
calibration signal. The α that minimizes this mixed score wins.

Compared to exp-017's gate, this is a continuous trade rather than binary
accept/reject. It can also *advance* an α that isn't the cal-best if it's
much better on held-out — something the gate can't do.

## Setup

Identical to exp-017 except for the `cv_strategy="mixed", cv_weight=2.0`
flags to `awq.calibrate`. Same model, quants, holdout chunk, comparison set.

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/build_holdout_chunk.py   # once
PYTHONPATH=src .venv/bin/python scripts/run_exp018_awq_cv_mixed.py
```

Read results:

```bash
cat out/exp-018/google__gemma-4-31B-it/table.md
```

Smoke-test:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_toolcall.py \
  --model out/exp-018/google__gemma-4-31B-it/IQ2_M-awq.gguf \
  --holdout out/smoke/holdout.jsonl \
  --out out/smoke/exp-018-IQ2_M.csv \
  --temperature 0.0 --ctx 8192 --seed 1000
```

## Success criteria

- **Held-out tool-call accuracy > exp-010 baseline (0.40)** — if mixed loss
  finds α values that genuinely generalize better than the group α, we
  should beat both exp-010 and exp-017.
- **PPL between FP16 (302) and exp-010 (2407)** for IQ2_M — i.e. some
  improvement over naive AWQ but no PPL collapse signaling over-fit.
- **KLD ≤ exp-010 baseline** — held-out weighting should pull α toward
  FP16-fidelity-preserving choices.

If both exp-017 and exp-018 fail to beat exp-010, per-tensor α just isn't
worth the complexity for sub-3-bpw Gemma; close the chapter.
