# Experiment 007: Quant calibration sweep on google/gemma-4-E4B-it

- **Status:** done
- **Created:** 2026-05-31
- **Branch:** `exp/007-quant-calibration-gemma` _(create with `git checkout -b exp/007-quant-calibration-gemma`)_

## Summary

Compare two 4-bit GGUF quant types (Q4_K_M and IQ4_NL) across five
calibration corpora on `google/gemma-4-E4B-it`. F16 acts as the
reference. All KLD/PPL numbers are produced at `ctx=4096` against the
same `corpus.eval.txt` (the 50k-token tool-call holdout used in
exp-001), and KLD is taken against the F16 logit dump at that context.

The five imatrix corpora are reused verbatim from exp-001:

- `wiki`             — `wiki.test.raw` (wikitext-2-raw-v1), ctx=512
- `custom`           — `logtrain.jsonl` + `calibration_supplement.txt`, ctx=512
- `mixed512`         — 500k custom tokens + full wiki, **imatrix ctx=512**
- `mixed2k`          — 500k custom tokens + full wiki, **imatrix ctx=2048**
- `mixed8k`          — 500k custom tokens + full wiki, **imatrix ctx=8192**

## Quantization Test with calibration data using google/gemma-4-E4B-it

_Filled in by `scripts/run_exp007_quant_calibration_gemma.py` (table
written to `out/exp-007/table.md`)._

| quant | technique | dataset | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |
|---|---|---|---|---|---|---|---|
| FP16   | none    | —             | 14.02 | 16.018 | 4.7841 | 0.00000 | 100.0000 |
| Q4_K_M | imatrix | wiki.test.raw | 4.97 | 5.677 | 4.8114 | 0.03959 | 94.2930 |
| Q4_K_M | imatrix | custom | 4.97 | 5.677 | 4.8479 | 0.03777 | 94.5020 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=512) | 4.97 | 5.677 | 4.8315 | 0.03891 | 94.2710 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=2048) | 4.97 | 5.677 | 4.8223 | 0.03786 | 94.3730 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=8192) | 4.97 | 5.677 | 4.8342 | 0.03710 | 94.5510 |
| IQ4_NL | imatrix | wiki.test.raw | 4.84 | 5.527 | 4.7534 | 0.04689 | 93.8760 |
| IQ4_NL | imatrix | custom | 4.84 | 5.527 | 4.7658 | 0.04464 | 93.9870 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=512) | 4.84 | 5.527 | 4.7953 | 0.04469 | 93.9820 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=2048) | 4.84 | 5.527 | 4.7685 | 0.04521 | 93.9070 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=8192) | 4.84 | 5.527 | 4.7527 | 0.04556 | 93.8490 |

Direction: lower is better for PPL and KLD; higher is better for same_top_p.

## Observations

- **IQ4_NL is smaller and lower-PPL than Q4_K_M** on every cell — 4.84
  GiB / 5.527 BPW vs 4.97 GiB / 5.677 BPW, and PPL ~4.75-4.80 vs
  ~4.81-4.85. The PPL gap (~0.06) is consistent across all 5 corpora.
- **But Q4_K_M wins on distribution-shape metrics**: KLD is ~16-20%
  lower (0.037-0.040 vs 0.045-0.047) and same_top_p is 0.3-0.7 pts
  higher across the board. Same story as exp-001 (`none` vs imatrix):
  PPL and KLD/same_top_p disagree about which artifact is "closer" to
  F16. IQ4_NL gets a slightly better average log-loss on this eval set
  while diverging more from F16's output distribution.
- **Corpus barely matters within either quant type.** Within Q4_K_M:
  PPL spread ≤ 0.037, KLD spread ≤ 0.0025, same_top_p spread ≤ 0.28
  pts across the 5 corpora. Within IQ4_NL: PPL spread ≤ 0.043, KLD
  spread ≤ 0.0023, same_top_p spread ≤ 0.14 pts. Same saturation
  pattern as the Qwen-family rows in exp-001.
- **Imatrix context length sweep (mixed512 / mixed2k / mixed8k) is
  flat in both quants** — random ordering inside noise. Confirms the
  exp-001 follow-up finding on a second quant type.
- **Pick by what you optimize for**: IQ4_NL if you want smaller files
  + lower raw PPL; Q4_K_M if you want minimum divergence from F16's
  output distribution (which is what matters for sampling-stability
  and task-level behavior, per the exp-001/002 pattern). Task-level
  eval on the IQ4_NL rows would resolve which side of the
  PPL/KLD tradeoff actually moves downstream metrics — open follow-up.

## Caveats

- Gemma's ~262k vocab busts `llama-perplexity`'s logit-vector
  allocation at `ctx=8192`, so this experiment (like the gemma rows in
  exp-001) runs the KLD baseline + per-quant bench at `ctx=4096`.
- Q4_K_M numbers are reused from exp-001 (built at the same
  `eval_ctx`); IQ4_NL numbers are new in this experiment.
- All quant files are roughly the same size — the actual GiB / BPW
  reported per row will differ slightly between Q4_K_M and IQ4_NL
  because of how the type packs sub-byte weights.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/run_exp007_quant_calibration_gemma.py
```

Prerequisites (all produced by exp-001's gemma run): `model-f16.gguf`,
`baseline.kld`, `corpus.eval.txt`, and `imatrix-{custom,wiki,mixed512,mixed2k,mixed8k}.gguf`
under `out/exp-001/google__gemma-4-E4B-it/`.
