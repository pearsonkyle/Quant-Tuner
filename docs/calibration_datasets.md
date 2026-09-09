# Calibration datasets

The pipeline consumes **seven** calibration sources by default (see
`UniversalConfig.sources`). This page documents each one, what it is for, and
how to point the pipeline at a pre-built corpus.

## The seven sources

| Source | What it is | Default budget | Role |
|---|---|---|---|
| `logs` | on-disk CLI usage logs + harvested agent trajectories (`datasets/agent-logs/data/`) | 2M tokens | tool-calling structure, real tool schemas, the harness's system prompt |
| `reasoning` | windows cut so a reasoning turn lands **last** (the only position a chat template keeps reasoning in) | 1M tokens | the model's dominant output mode — without this the corpus carries ~no reasoning |
| `swe-trajectories` | [`pearsonkyle/swe-agentic-trajectories`](https://huggingface.co/datasets/pearsonkyle/swe-agentic-trajectories) `resolved` split (71 verified SWE-rebench solutions) | 1M tokens | long agentic sessions, 19 languages, ~34 tool calls each |
| `broad-supplement` | [`pearsonkyle/broad-domain-supplement`](https://huggingface.co/datasets/pearsonkyle/broad-domain-supplement) `corpus` split | all of the `calib` half | breadth (the `mtp` half is reserved for draft-head training) |
| `llmtk-sft` | [`pearsonkyle/llmtk-sft-corpus-v2`](https://huggingface.co/datasets/pearsonkyle/llmtk-sft-corpus-v2) `calibration-15m-v65536-ctx32k` slice | 4M tokens (of 15M) | the agentic-coding SFT distribution — the slice the 32K-ctx recipes are calibrated on |
| `redteam-refusals` | [`pearsonkyle/redteam-safety-disclosures`](https://huggingface.co/datasets/pearsonkyle/redteam-safety-disclosures) refused rows | all of them | refusal behavior (attack prompts + generic refusals; the targets' original completions never reach a corpus) |
| `wiki` | `out/exp-001/wiki/wiki.test.raw` (Wikipedia test split) | all of it | general prose |

Each source is **split by a stable group key first**, then interleaved
proportionally (never concatenated) into `corpus.cal.txt`. A token-budgeted
calibrator (AWQ, GPTQ) samples a fixed slice of the finished file, so a source
written as one contiguous block either eats the budget or misses it — the
interleave is what keeps every source present in a sampled slice.

## The 32K-ctx slice (llmtk SFT)

The 32K-ctx recipes (`src/quant_tuner/recipes/*_32k.yaml`) are calibrated on a
corpus where **every window is a full 32K context** — not 2 048-token fragments
glued together. Two invariants make that true:

1. **The upstream corpus is pre-windowed at 65 536.** The
   `calibration-15m-v65536-ctx32k` slice arrives as one string per
   conversation, already rendered with a chat template. The builder re-chunks
   it to the calibrator's actual ctx (32 768 for the 32K recipes) so a window
   never straddles a boundary.
2. **`ctx` is a packing parameter, not a runtime flag.** `UniversalConfig.ctx`
   is the context every calibrator reads the corpus at. It is passed to
   `llama-imatrix -c`, `awq.calibrate(ctx=)`, and `gptq.calibrate(ctx=)` — and
   the corpus windows are packed to fill one `ctx` exactly. A window longer
   than `ctx` straddles a chunk boundary (the calibrator sees two unrelated
   conversations in one context); a window much shorter than `ctx` means the
   calibrator sees two unrelated conversations glued together.

### Building a 32K-ctx corpus

```bash
uv run python scripts/build_universal_corpus.py \
    --out out/<run>/corpora \
    --model out/<run>/model_extracted \
    --ctx 32768 \
    --cal-llmtk-sft-tokens 4000000
```

That writes `corpus.cal.llmtk_sft.txt` (the per-source intermediate) and
interleaves it into `corpus.cal.txt`. The 32K recipes then point at the
pre-built file via `data.corpus`:

```yaml
data:
  corpus: ./out/<run>/corpora/corpus.cal.txt
  # `logs` is no longer required — the pipeline reuses the pre-built file
```

### A local override

If you have the slice on disk (one string per record, `\n\n`- or `\n`-delimited):

```bash
uv run python scripts/build_universal_corpus.py \
    --out out/<run>/corpora \
    --model out/<run>/model_extracted \
    --llmtk-sft /path/to/calibration-15m-v65536-ctx32k.txt
```

## Eval corpora (each gets its own baseline)

The eval side is **separate from calibration** by construction — a winning
calibration that over-fits the eval slice would conflate fit with
generalization. Each eval corpus is a different distribution and **must get its
own `baseline.kld`**:

| Corpus | Source | What it measures |
|---|---|---|
| `corpus.eval.txt` | external `eaddario/imatrix-calibration` (code + math + tools) | headline PPL/KLD — neither cal nor val appears here |
| `corpus.eval.general.txt` | external `combined_en_tiny` | broad English |
| `corpus.eval.tools.txt` | on-disk logs **holdout** (10%) | in-distribution tool-call PPL (quant-vs-quant, not absolute) |
| `corpus.eval.agentic.txt` | SWE trajectories **holdout** (10% of `resolved`) | long agentic sessions |
| `corpus.eval.broad.txt` | broad-supplement `mtp` half eval slice | breadth |
| `corpus.eval.redteam.txt` | held-out red-team attacks + refusals | refusal behavior |

⚠️ `llama-perplexity` has no `--parse-special`, so a chat-templated eval slice
tokenizes control markers as plain BPE — a distribution the model never sees.
Use `corpus.eval.tools.txt` / `.agentic.txt` for **quant-vs-quant** comparison
(e.g. the windowed-packer A/B), not absolute PPL.

## Disjointness invariant

`build()` asserts at the end that **no row appears in both a calibration
corpus and an eval corpus** (per source, by stable group key). A failure here
means a split leaked — do not ship a calibration built from a corpus that
failed this check.
