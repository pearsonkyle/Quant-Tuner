---
license: other
task_categories:
- text-generation
tags:
- sft
- instruction-tuning
- agentic
- tool-use
- qat
- distillation
- private
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: holdout
    path: data/holdout.jsonl
  - split: test
    path: data/test.jsonl
---

# Universal SFT Mixture

Full conversations from every source the calibration pipeline uses — CLI logs, agent trajectories, verified SWE solutions, broad-domain instructions and refusal pairs — one schema, split-tagged to match the calibration corpus.

**Version `0.1.0`** · built 2026-08-15T08:58:48

| split | rows | verified (tests pass) | mean tool calls | size |
| --- | ---: | ---: | ---: | ---: |
| `train` | 6170 | 0 | 4.3 | 85.2 MB |
| `holdout` | 406 | 0 | 9.8 | 11.0 MB |
| `test` | 67 | 0 | 47.6 | 11.4 MB |

## ⚠️ Private by design

This dataset carries **real captured usage** — interactive CLI coding sessions and agent
trajectories, ~91% of it by character count. Those logs contain prompts, file contents and
paths from actual work and are not ours to publish. The registry marks this dataset
`private_only`, so `scripts/dataset.py push` **refuses a public upload** rather than relying
on anyone remembering `--private`. Keep the repo private. Do not mirror it.

## What one row is

One complete conversation, exactly as it happened, with nothing trimmed: no windowing, no
tool-output clipping, no system/schema stubbing, and no chat template applied. `tool_calls`
and `reasoning_content` stay as separate message fields so a trainer can render them through
whichever template it wants.

## Sources

| `source` | what it is |
| --- | --- |
| `logs` | interactive CLI coding sessions (Claude Code / opencode / qwen code) |
| `logs-agents` | harvested agent trajectories, 19 languages, tests-verified |
| `swe-trajectories` | verified SWE-rebench solver runs — real issues, hidden tests passed |
| `broad-instruct` | the broad-domain supplement's instruction view |
| `redteam-refusals` | red-team attack prompts paired with **generic refusals** |

The refusal rows never carry what a target model actually said. Every assistant turn is
replaced from a deterministic refusal bank; the original completions and `target_reasoning`
do not reach this file. Refusal behaviour is what low-bit quantization erodes first, so the
attack *distribution* belongs in training — the harmful responses do not.

## The split is an eval boundary, not a ratio

`split` mirrors the calibration corpus's own assignment, so training on `train` leaves the
tools / agentic / breadth / refusal PPL-KLD eval holdouts genuinely held out. It is not a
conventional train/validation ratio and should not be reshuffled — re-splitting it randomly
would put eval rows into training and quietly invalidate every downstream number.

## Using it

```python
from datasets import load_dataset

ds = load_dataset("pearsonkyle/universal-sft-mixture", split="train")
agentic = ds.filter(lambda r: r["source"] in ("logs-agents", "swe-trajectories"))
```

Straight into the repo's QAT path, which is what this file is the input to:

```bash
PYTHONPATH=src python scripts/build_sft_qat_corpus.py \
    --sft out/corpora/qwen3-universal-v2/sft.jsonl.gz \
    --window 8064 --max-tool-tokens 3072 --min-density 0.05 \
    --out out/sft_corpus_8064.pt
```

Scale `--max-tool-tokens` with the window (3072 at 8064, 4096 at 12288). 1024 was only ever
right at a 4096 window and drops 28% of all conversation content.

## Caveats

* **System prompts are scrubbed, not anonymized.** Repeated agent-harness boilerplate is
  dropped; a repeated block is *kept* when it names a path this conversation actually
  touches. That is a relevance filter, not a privacy control — real paths and file contents
  remain throughout.
* **Heavily imbalanced by length.** `broad-instruct` is 83% of the rows and 5% of the
  characters; the two log sources are 10% of rows and 91% of characters. Budget per source
  when training, or the logs dominate.
* Tool outputs are raw container/shell stdout and can be very long.
* Reasoning survives a render only on assistant turns after the last user turn — the whole
  conversation for an agentic trajectory, only the tail for a multi-turn chat log.
* Token counts are off by default in the build (they cost a second pass and are specific to
  whichever tokenizer built it), so rows carry `n_chars` rather than `n_tokens`.

## Row schema

| field | meaning |
| --- | --- |
| `id` | stable id of the source conversation |
| `source` | which corpus it came from (see the table above) |
| `split` | `train` / `holdout` / `test` — **an eval boundary, not a ratio** |
| `messages` | the full conversation; `tool_calls` and `reasoning_content` are separate message fields, no template applied |
| `tools` | the tool schemas the conversation was conditioned on, where it had any |
| `meta` | per-source extras (repo/instance for SWE, area/register for breadth) |
| `n_messages`, `n_tool_calls`, `n_tool_results`, `n_reasoning`, `n_chars` | shape |

## Reproducing

Generated with [Quant-Tuner](https://github.com/pearsonkyle/Quant-Tuner); see
`docs/ternary_qat.md` for the end-to-end pipeline and
`src/quant_tuner/datasets/` for the exact builder used to publish this.
