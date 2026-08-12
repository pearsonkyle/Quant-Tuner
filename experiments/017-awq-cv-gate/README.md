# Experiment 017 — Per-tensor α with held-out KLD gate

> **⚠️ Superseded by exp-019. Do not use these numbers for comparison.**
>
> The cal and held-out corpora here were both drawn from the same logtrain
> distribution (`exp-009/corpus.mixed8k.txt` and
> `out/holdout_chunks/cv_1k.txt`), so the held-out signal was a re-draw of
> the calibration distribution rather than a generalization probe. The
> bench eval corpus also overlapped calibration. exp-019 reruns this
> technique with disjoint cal/val/eval corpora. The big GGUF artifacts
> (`*.gguf`, `model_awq/`) have been deleted to free disk; `awq.pt`,
> `results.csv`, `table.md`, and `logs/` are retained for reference.

## Hypothesis

exp-016 (per-tensor α refinement) collapsed PPL on the calibration eval text
(2407 → 30) but tool-call accuracy on a held-out set *dropped* (40% → 25%
tool selection). The α grid (±0.25 around group α) was selecting per-member
scales that fit the calibration distribution at the cost of FP16 fidelity
and held-out task performance.

This experiment keeps the per-tensor α refinement but adds a **binary gate**:
each per-member α is accepted only if it doesn't worsen proxy reconstruction
loss on a held-out activation chunk (logtrain test slice, disjoint from
calibration AND from the tool-call smoke holdout). When the calibration-best
α loses on held-out, it reverts to the group α — a safe per-group fallback.

## Setup

- Model: `google/gemma-4-31B-it`
- Quants: `IQ2_XS`, `IQ2_M`, `Q2_K_S`
- Reuses exp-009 artifacts (model, F16 GGUF, calibration corpus, imatrix,
  eval corpus, baseline KLD)
- New input: `out/holdout_chunks/cv_1k.txt` (logtrain test slice sessions
  5-24, disjoint from calibration and from `out/smoke/holdout.jsonl`)
- Comparison vs exp-009 / exp-010 / exp-016 per quant

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/build_holdout_chunk.py   # once
PYTHONPATH=src .venv/bin/python scripts/run_exp017_awq_cv_gate.py
```

Read results:

```bash
cat out/exp-017/google__gemma-4-31B-it/table.md
```

Smoke-test held-out tool-call accuracy on IQ2_M:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_toolcall.py \
  --model out/exp-017/google__gemma-4-31B-it/IQ2_M-awq.gguf \
  --holdout out/smoke/holdout.jsonl \
  --out out/smoke/exp-017-IQ2_M.csv \
  --temperature 0.0 --ctx 8192 --seed 1000
```

## Success criteria

- **Held-out tool-call accuracy ≥ exp-010 baseline** (0.40 tool selection on
  the 3-session smoke). The whole point is to undo exp-016's regression.
- **PPL not collapsed below FP16** (302) — if PPL is in the same ballpark
  as exp-010 (2000–3000), that's a sign the gate is correctly preventing
  over-fit perturbations.
- **F16 sanity drift back to exp-010 levels** (~0.10–0.15), since most
  per-member α refinements should revert to the group α.

If gate works: this becomes the safe default. If gate too aggressively
reverts everything, exp-018 (mixed loss, w=2) tries a softer version.
