---
license: other
task_categories:
- text-generation
tags:
- calibration
- agentic
- sft
- coding
- tool-use
size_categories:
- n>1M
configs:
- config_name: default
  data_files:
  - split: calibration-15m-v65536-ctx32k
    path: data/calibration-15m-v65536-ctx32k.txt
---

# llmtk-sft corpus (32K-ctx calibration slice)

**Not a publishable dataset — the payload lives on the Hub, not here.** The card
documents what `quant-tuner` consumes from
[`pearsonkyle/llmtk-sft-corpus-v2`](https://huggingface.co/datasets/pearsonkyle/llmtk-sft-corpus-v2)
and how to use it for calibration.

**Version `0.1.0`**

## What it is

A 15M-token SFT corpus of agentic coding conversations, pre-rendered into plain
text and pre-packed at a 65 536-token context. The split the repo consumes is
`calibration-15m-v65536-ctx32k` — a slice of that corpus that is:

* **Agentic in shape.** Every conversation carries real tool calls, tool results,
  multi-step reasoning, and the long-context conditioning that a coding agent
  sees mid-session (5k–288k tokens per session in the upstream corpus).
* **Plain text.** The upstream has already applied a chat template; the records
  arrive as one string per conversation. No tool schema, no `messages` field,
  no `reasoning_content` field — just the rendered text the model would emit
  and condition on.
* **Pre-windowed at 65 536.** The records are cut so each one fits in one
  long-context window. Our builder re-chunks them to the calibrator's actual
  ctx (default 8 192; the 32K recipes use 32 768) so a calibrator reading the
  corpus at ctx N never sees a window that straddles a boundary.

## Why it is a separate source, not folded into `logs`

The on-disk log corpora (`datasets/agent-logs/data/logs-*.jsonl.gz`) are the
**chat-shaped** rows the SWE-rebench / tool-call evals were built against —
they carry `messages`, `tools`, and the original harness's system prompt. The
llmtk-sft slice is the **rendered** text of a different (larger, agentic-coding-
focused) corpus. Treating it as its own source lets the builder:

1. Apply its own token budget independently of the logs.
2. Pack it with `pack_raw_samples` (shuffled, windowed) rather than the
   chat-templated `stratified_pack` — the two packers are tuned for different
   input shapes.
3. Show up as its own row in `token_share` in `corpora_audit.json`, so a
   regression that drops it is visible.

## Row schema (what the builder sees)

One record per line / blank-line pair, each a plain string of rendered
conversations. No structured fields.

| field | meaning |
| --- | --- |
| (the string itself) | one or more rendered chat conversations, chat-template applied, tool calls inlined as the model sees them |

## Using it in a recipe

The simplest path is to let the universal corpus builder pull it from the Hub:

```bash
uv run python scripts/build_universal_corpus.py \
    --out out/<run>/corpora \
    --model out/<run>/model_extracted \
    --ctx 32768 \
    --cal-llmtk-sft-tokens 4000000
```

That writes `corpus.cal.llmtk_sft.txt` (the per-source intermediate) and
interleaves it into `corpus.cal.txt`. The 32K-ctx recipes under
`src/quant_tuner/recipes/*_32k.yaml` set `data.corpus: <that file>` so the
pipeline consumes the pre-built corpus directly.

A local override (one string per record, `\n\n`- or `\n`-delimited):

```bash
uv run python scripts/build_universal_corpus.py \
    --out out/<run>/corpora \
    --model out/<run>/model_extracted \
    --llmtk-sft /path/to/calibration-15m-v65536-ctx32k.txt
```

## Caveats

* **Not chat-shaped.** Because the records are already rendered, the builder
  cannot re-apply the target model's own chat template. If your target model's
  template differs materially from the upstream one (e.g. a different tool-call
  delimiter), the calibrator will see text the model has never rendered. For
  Qwen-family targets (the intended use) the upstream template is the stock
  Qwen3.5/3.6/3.8 one, so this is a non-issue.
* **Eval contamination.** The llmtk-sft slice is a **calibration** source. The
  PPL/KLD eval corpora (`corpus.eval.*.txt`) are built from the external
  `eaddario/imatrix-calibration` set, the on-disk logs holdout, and the
  SWE/broad/redteam holdouts — none of which overlap with this slice. Do not
  add llmtk-sft rows to an eval corpus.
* **15M tokens is a lot.** The default 4M budget in `UniversalConfig` takes a
  quarter of the slice. Raise `--cal-llmtk-sft-tokens` if you want the mix to
  lean harder on agentic-coding text; the cost is llama-imatrix wall-clock
  (linear in tokens).
