# Experiment 001: imatrix with custom data

- **Status:** done
- **Tests hypothesis #1:** investigate the imatrix technique and see if we can improve it using custom data or a new optimization metric
- **Created:** 2026-05-25T17:05:55+00:00
- **Branch:** `exp/001-imatrix-with-custom-data` _(create with `git checkout -b exp/001-imatrix-with-custom-data`)_

## Summary

For each of three Qwen3.5-class 9B models, build three Q4_K_M GGUFs and
measure them against the same held-out eval set. The only thing that varies
across the three GGUFs for a given model is **what the importance matrix
was built from** (or whether one was used at all):

1. `custom` — `llama-imatrix` on `logtrain.jsonl` + `calibration_supplement.txt`
2. `wiki`   — `llama-imatrix` on `wiki.test.raw` (wikitext-2-raw-v1)
3. `none`   — `llama-quantize` with no imatrix

Re-weighting variant is **vanilla llama.cpp** (no `calibrate.imatrix` pass).
That keeps this experiment about *the corpus*; exp-002 will hold the corpus
fixed and vary the re-weighting variant.

## Approach

- **Models** (HF repo ids used verbatim):
  - `Qwen/Qwen3.5-9B`
  - `Tesslate/OmniCoder-9B`
  - `Jackrong/Qwopus3.5-9B-Coder`
  - `google/gemma-4-E4B-it` (added later; KLD baseline + bench run at `ctx=4096`
    because Gemma's ~262k vocab busts `llama-perplexity`'s logit-vector
    allocation at `ctx=8192`. Within-gemma comparison is valid; absolute
    KLD/PPL aren't directly comparable to the Qwen rows above.)
- **Pipeline per model** (one F16 GGUF, one KLD baseline, shared across all 3 cells):
  1. `extract.extract_text_lm` → text-only HF dir
  2. `convert.hf_to_f16_gguf` → F16 GGUF
  3. `prepare_corpora` → custom train corpus (`logtrain.jsonl` + supplement) + eval corpus (holdout slice)
  4. `llama_cpp.imatrix(f16, custom_corpus)` → `imatrix-custom.gguf`
  5. `llama_cpp.imatrix(f16, wiki.test.raw)` → `imatrix-wiki.gguf`
  6. `kld.build_baseline(f16, eval_corpus)` → F16 KLD reference (once per model)
  7. Three `gguf.quantize` calls (custom / wiki / none) → three Q4_K_M GGUFs
  8. `bench.runner.bench_one(..., suite="kld")` per quant → one CSV row each
- All stages are wrapped in `experiments.step()` so reruns skip completed work.

## Metrics

Rendered from `out/exp-001/results.csv` by
`scripts/render_exp001_table.py`. Filled in after the run.

| model | technique | dataset | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |
|---|---|---|---|---|---|---|---|
| Jackrong/Qwopus3.5-9B-Coder | imatrix | custom        | 5.24 | 5.029 | 3.3549 | 0.51300 | 90.4960 |
| Jackrong/Qwopus3.5-9B-Coder | imatrix | wiki.test.raw | 5.24 | 5.029 | 2.9200 | 0.53350 | 90.4180 |
| Jackrong/Qwopus3.5-9B-Coder | imatrix | 500k-custom+wiki (ctx=8192) | 5.24 | 5.029 | 3.1578 | 0.51996 | 90.3440 |
| Jackrong/Qwopus3.5-9B-Coder | none    | —             | 5.24 | 5.029 | 2.5144 | 0.95961 | 87.6340 |
| Qwen/Qwen3.5-9B             | imatrix | custom        | 5.24 | 5.029 | 3.8167 | 0.69968 | 88.7470 |
| Qwen/Qwen3.5-9B             | imatrix | wiki.test.raw | 5.24 | 5.029 | 3.4109 | 0.71204 | 88.8550 |
| Qwen/Qwen3.5-9B             | none    | —             | 5.24 | 5.029 | 3.2525 | 1.12402 | 86.0420 |
| Tesslate/OmniCoder-9B       | imatrix | custom        | 5.24 | 5.029 | 3.9426 | 0.70067 | 88.9130 |
| Tesslate/OmniCoder-9B       | imatrix | wiki.test.raw | 5.24 | 5.029 | 3.4310 | 0.70769 | 88.8990 |
| Tesslate/OmniCoder-9B       | none    | —             | 5.24 | 5.029 | 3.2500 | 1.12088 | 86.0710 |
| google/gemma-4-E4B-it †     | imatrix | custom        | 4.97 | 5.677 | 4.8479 | 0.03777 | 94.5020 |
| google/gemma-4-E4B-it †     | imatrix | wiki.test.raw | 4.97 | 5.677 | 4.8114 | 0.03960 | 94.2930 |
| google/gemma-4-E4B-it †     | imatrix | 500k-custom+wiki (ctx=8192) | 4.97 | 5.677 | 4.8342 | 0.03710 | 94.5510 |
| google/gemma-4-E4B-it †     | none    | —             | 4.97 | 5.677 | 5.0167 | 0.10911 | 90.8470 |

† Gemma rows were evaluated at `ctx=4096` (vs `ctx=8192` for everything above);
absolute PPL/KLD aren't directly comparable across that boundary.

Direction: lower is better for PPL and KLD; higher is better for same_top_p.

## Observations

- All 9 Qwen-class cells produced GGUFs of identical size (5.24 GiB) and BPW
  (5.029); the 4 gemma cells likewise all land at 4.97 GiB / 5.677 BPW —
  Q4_K_M parameter choice is invariant to the imatrix corpus, as expected.
  (The BPW gap between families reflects Gemma's tied embedding + much
  larger vocab, not the calibration.)
- **PPL and KLD disagree.** `none` has the lowest (best) PPL on every model,
  but the worst KLD (by ~60%) and the worst same_top_p (~2.5 pts behind both
  imatrix cells).
- Within `imatrix`, the **wiki** corpus produced lower PPL than the **custom**
  corpus on all three models, but KLD and same_top_p were essentially
  identical between the two corpora (deltas ≤ 0.02 KLD, ≤ 0.1 pts same_top_p).
- All three models track each other closely on the imatrix rows. OmniCoder
  and Qwen3.5-9B in particular are nearly indistinguishable on every metric —
  consistent with OmniCoder being a fine-tune of the base.
- The eval set is 50k tokens of held-out tool-call sessions; PPL on this
  distribution is dominated by structured / repetitive tokens, which may
  explain why the uncalibrated quant scores well on average log-loss while
  diverging most from F16 in distribution shape.
- **Gemma flips the PPL story.** On `google/gemma-4-E4B-it`, `none` is the
  *worst* cell on both PPL (5.017 vs ~4.83 for the three imatrix cells) and
  KLD (0.109 vs ~0.038 — almost 3×), and trails the imatrix cells by ~3.7
  pts on `same_top_p`. Within imatrix, the three corpora cluster tightly
  again (PPL spread ≤ 0.04, KLD spread ≤ 0.002, same_top_p spread ≤ 0.26
  pts). `mixed8k` edges out the others on KLD and same_top_p, `wiki` on
  raw PPL. The qualitative ordering "imatrix ≫ none, corpus barely
  matters" survives the cross-architecture jump; the magnitude is just
  much larger here.
- **Follow-up on Jackrong only:** a `mixed8k` cell built an imatrix from
  500k custom tokens concatenated with the full wiki.test.raw at
  `ctx=8192` (vs. `ctx=512` for the original cells). KLD (0.520) and
  same_top_p (90.344) land in the same tight cluster as `custom` and
  `wiki`; PPL (3.158) falls between them. Reinforces the
  "corpus + context length don't move distribution-shape metrics much
  once you're past saturation" reading.

## Analysis

- **KLD and same_top_p tell one story; PPL tells a different one.** On these
  models and this eval set, calibration with an imatrix lowers KLD by ~40%
  and lifts top-token agreement with F16 by ~2.5 pts, but slightly raises
  PPL. Read this as: imatrix preserves the *shape* of F16's output
  distribution (which is what matters downstream — sampling, tool-call
  decisions) while no-calibration finds a quant that happens to assign
  slightly higher mass to the eval-set tokens on average. PPL alone would
  point you to the wrong artifact.
- **Corpus didn't matter much.** Wiki vs. custom for the imatrix differ by
  ≤ 2% on KLD and ≤ 0.15 pts on same_top_p across all three models. The
  500k-token custom corpus didn't out-calibrate generic wikitext on this
  eval. Two possible explanations: (a) the eval set is too narrow / too
  similar across cells for corpus differences to surface; (b) vanilla
  `E[a²]` re-weighting may be coarse enough that the *which corpus*
  question is dominated by the *how many tokens* question, and both
  corpora are above whatever saturation point applies.
- **Method-vs-corpus question is open.** This experiment held the variant
  fixed at vanilla; exp-002 will hold the corpus fixed and vary the
  variant. If hybrid_custom moves KLD/same_top_p meaningfully on the
  custom corpus but not on wiki (or vice versa), that's the interesting
  cross-term.

## Next steps

This experiment is closed — see exp-002 (output-aware + outlier
variants + tool-call eval) for the resolution of the open questions
below. Tracked-forward items:

- **Exp-002 (done)**: the variant axis was tested on Jackrong only.
  `hybrid_custom` was refuted; `outlier_l4` won every distribution-shape
  metric and ties `none` on schema_valid pass@5. The original "exp-002
  will hold the corpus fixed and vary the variant" plan was executed,
  but the winning variant turned out to be `outlier_l4`, not
  `hybrid_custom` (which is what `q4_k_m_imatrix.yaml` currently uses).
- **Generalization across models is still open**: exp-002's variant
  comparison ran on Jackrong only. The exp-001 finding that
  "corpus barely matters" was 3-model, so we have cross-model coverage
  there. But "outlier_l4 wins" is currently 1-model. Pending follow-up
  in exp-002's Next steps section.
- **Task-level signal is no longer deferred** — it was added in exp-002
  and changed the verdict for `output_aware`. The fact that
  `none` ties or beats every imatrix variant on `schema_valid` mean was
  the single most consequential finding of this entire research thread.
- **Confound to investigate**: the per-model PPL gap between custom and
  wiki imatrix is suspiciously consistent (~0.4–0.5 across all three
  models). Worth a 5-minute look at the imatrix files themselves
  (`llama-imatrix --show-stats`) to confirm the custom corpus produced
  the expected token coverage.
