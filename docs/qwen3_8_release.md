# exp-060: calibrated GGUF ladder for Qwen3.8-27B (+ MTP)

Runbook for standing up a new model release end to end: **IQ2_M · IQ3_M · IQ4_XS · Q5_K_M**,
each with the MTP draft head bundled at Q8_0, calibrated on the universal corpus (every
dataset in `datasets/`, plus raw wiki).

`Qwen/Qwen3.8-27B` is unreleased at the time of writing. Every stage below is exercisable
today against a released sibling — pass `--model Jackrong/Qwopus3.6-27B-Coder --run
exp-060-dryrun` and the whole chain runs unchanged.

## What's new relative to the exp-041 (Qwopus3.6) release

| | exp-041 | exp-060 |
|---|---|---|
| Calibration sources | CLI logs + wiki | + **multi-language agent trajectories**, **SWE trajectories**, **broad-domain supplement**, **red-team attacks paired with refusals**, and reasoning-terminal windows |
| Chat template | assumed | **checked against a tool-calling fixture before anything is built**, and the built corpus is re-scanned for markers |
| Reasoning | ignored (≈none survived templating) | **normalized across source shapes and given its own window cut**, with coverage measured |
| MTP pin | hardcoded `blk.64.` | **read from the GGUF** (`models.mtp.describe`) |
| Eval corpora | external + general | + in-distribution **tools**, **agentic**, **broad** and **refusal** holdouts |

## Stage 0 — is the template usable? (seconds, no GPU)

```bash
uv run python scripts/verify_chat_template.py --model Qwen/Qwen3.8-27B --show
```

Everything downstream is chat-templated text, so this runs first. It renders a fixture with
two tools in scope, an assistant turn carrying prose *and* a call, a tool result, and a
closing turn, then asserts the schemas, the argument JSON and the result all survive, that
in-text markers still tokenize to single ids, and that the packer's window slicing doesn't
come back empty. A blocking failure means the corpus would contain no tool-call structure —
which produces a quant that benches fine and can't tool-call.

## Stage 1 — fetch, keep the MTP head, convert

```bash
PYTHONPATH=src .venv/bin/python scripts/exp060_setup_qwen38.py
```

* Cross-checks `config.json`'s draft-layer count against the **actual `mtp.*` weights**.
  Only "declared **and** present" turns on `keep_mtp` — Ornith-1.0-9B declared a head it
  didn't ship, and keeping it produced a phantom block that crashed llama.cpp.
* Converts to F16 and writes `out/exp-060/mtp_report.json` with the draft layer index and
  the `--tensor-type` pin that stage 2 consumes.
* If the model ships **no** head, the release can still go out; speculative decoding then
  needs a graft (`scripts/exp045_graft_mtp.py`, requires a byte-identical sibling) or a
  separate draft model. That is also the hook for a later distillation pass to train a
  better head.

## Stage 2 — the universal corpus

```bash
uv run python scripts/build_universal_corpus.py \
    --out out/exp-060/corpora --model out/exp-060/model_extracted
```

| Source | Cal slice | Eval slice |
|---|---|---|
| `datasets/agent-logs/data/logs-cli.jsonl.gz` (CLI usage logs) | train 80%, windowed | holdout 10% → `corpus.eval.tools.txt` |
| `datasets/agent-logs/data/logs-agents.jsonl.gz` (435 verified trajectories, 19 languages) | train 80%, windowed | holdout 10% → `corpus.eval.tools.txt` |
| reasoning-terminal windows (from the same train slice) | budgeted separately | — |
| `datasets/redteam-safety-disclosures` (attacks, **refused**) | 90% | 10% → `corpus.eval.redteam.txt` |
| `pearsonkyle/swe-agentic-trajectories` (resolved) | 90%, windowed | 10% → `corpus.eval.agentic.txt` |
| `pearsonkyle/broad-domain-supplement` (`calib` half) | raw text | 10% of the `mtp` half → `corpus.eval.broad.txt` |
| `wiki.test.raw` | chunked | — |
| `eaddario/imatrix-calibration` (external) | — | `corpus.eval.txt`, `corpus.eval.general.txt` |

Notes worth keeping in mind:

* Sources are **interleaved proportionally**, never concatenated — AWQ/GPTQ sample a fixed
  token budget across the file, and a contiguous block either eats the budget or misses it.
  Check `token_share` in `corpora_audit.json`; if wiki dominates, cap it with
  `--cal-wiki-tokens`.
* The supplement's **`mtp` half stays out of calibration** (bar its eval slice). It is
  reserved for MTP draft-head training — a head trained on text the trunk was calibrated on
  flatters its own acceptance rate.
* **Reasoning needs its own window shape.** Chat templates keep reasoning only on a render's
  final assistant turn, so ordinary windows carried 2 of 4,291 available reasoning turns into
  the corpus. `reasoning_windows` re-cuts the same conversations so a chain-of-thought turn
  lands last; the audit reports blocks available vs. blocks that landed. (Empty
  `<think></think>` blocks — which Qwen templates emit on every final turn — are not counted.)
* **Red-team rows are refused, never replayed.** Attack turns are kept verbatim, every
  assistant turn is replaced with a generic refusal from a small deterministic bank, and the
  original completions and `target_reasoning` never reach a corpus. Asserted on the built
  sessions, not just intended.
* Agentic tool outputs are head+tail clipped to `--max-tool-output-tokens` (512). A single
  raw pytest dump is 10k+ tokens and would consume a window on its own.
* The audit records the tool-call marker count **per source**. A source silently losing its
  calls is invisible in the total, which the other chat source keeps non-zero.

**How much data is there, and how much do we use?** Measured on the Qwen3.6 tokenizer
(message content, before windowing): logs-cli 14.2M + logs-agents 15.1M, swe-resolved 1.03M,
broad calib-half 0.34M, redteam-refused 0.09M, wiki 0.30M — ~31M tokens, overwhelmingly the
two log corpora. Broad, red-team and wiki are budgeted `None` = **all of them**; SWE is
data-bound (its cap is above what survives tool-output clipping). Only the logs are
budget-bound, at 2M — `--cal-logs-tokens` is the knob, and the cost of raising it is
llama-imatrix wall-clock, which is linear in corpus tokens.

Reference build at default budgets — **4.41M tokens, 9,403 tool calls, 5,942 tool results,
375 reasoning blocks**:

```
cal[logs]:            1,999,837 tokens (45.3%)  5,151 tool calls, 131 reasoning blocks
                      spread across 20 agent sources: qwen 15%, claude 13%, go 8%, cpp 8%, …
cal[reasoning]:       1,000,539 tokens (22.7%)  1,777 tool calls, 244 reasoning blocks
cal[swe-trajectories]:  680,418 tokens (15.4%)  2,475 tool calls   (all that survives clipping)
cal[broad-supplement]:  344,589 tokens ( 7.8%)      0 tool calls   (the whole calib half)
cal[redteam-refusals]:   88,812 tokens ( 2.0%)      0 tool calls   (all 348 attacks, refused)
cal[wiki]:              297,120 tokens ( 6.7%)      0 tool calls   (all of wiki.test.raw)
```

The log source is spread by the packer's `(source, length_bucket)` round-robin with a
per-stratum shuffle, so no single agent source dominates: 20 sources, none above 15%, 18 of
the 19 trajectory languages represented.

### Two extra outputs from the same pass

`corpus.cal.jsonl.gz` — the calibration corpus as records (`i`, `source`, `n_chars`,
`text`), one per window, byte-identical to `corpus.cal.txt`. This is what you query when a
number looks wrong and you need to know which source a given span came from.

`sft.jsonl.gz` — **the training view**, and deliberately not the calibration view. Every
chat-shaped source as complete conversations: no windowing, no tool-output clipping, no
system/schema stubbing, no chat template applied. `tool_calls`, `tool_call_id`/`name` and
`reasoning_content` are separate message fields, so a trainer can template them however it
likes and mask reasoning independently of the answer. `split` matches the calibration
split, so training on `split == "train"` leaves the tools / agentic / refusal holdouts
genuinely held out.

```
6,643 conversations · 85,643 messages · 33,572 tool calls · 4,289 reasoning turns · 87M chars
  broad-instruct    5,536      logs-agents  435      redteam-refusals  348
  logs (CLI)          253      swe-traj      71
```

**System prompts are scrubbed** in the SFT view (not in the calibration corpus, where the
packer already stubs prose). 90% of system-prompt characters in these logs sit in blocks
repeated verbatim across sessions — tone guidelines, git etiquette, worked examples — and
training on 24k tokens of preamble thousands of times teaches nothing. A repeated block is
**kept** when it names a path or file this conversation actually touches, which is what
separates "the repo layout" from "the harness". 6.4M → 0.4M characters; `--no-sft-scrub-system`
disables it. Neither frequency nor keywords alone works here: the harness blocks are full of
the words "repository" and "file paths", and generic filenames (`package.json`, `CLAUDE.md`)
plus library names (`Node.js`) ground nothing — they are filtered by document frequency.

**What the quantizers actually sample** is now reported and gated. AWQ and GPTQ don't read
the whole corpus; they stride a fixed budget across it. The audit runs the production sampler
over an index tensor and attributes every sampled token back to its source:

```
awq  (65k budget): 1.5% of corpus, 6 sources — logs 49% · reasoning 18% · swe 18% · broad 6% · wiki 6% · redteam 4%, 141 tool calls
gptq (32k budget): 0.7% of corpus, 6 sources — logs 50% · reasoning 17% · swe 12% · wiki 11% · broad 8% · redteam 2%,  59 tool calls
```

A build whose GPTQ slice contains zero tool calls now fails rather than shipping. The corpus
is also checked for control tokens on the bytes as written (`<tool_call>` 9,393,
`<tool_response>` 5,936, `<|im_end|>` 16,125) — every one must tokenize to a single id, since
`llama-imatrix --parse-special` and the HF-side calibrators both depend on that.

Pass `--sft-token-counts` for per-conversation `n_tokens` (a second tokenization pass over
~30M tokens, and specific to `--model`'s tokenizer); `--no-sft` skips the file.

### Calibration context: 8192, and why the corpus is packed for it

The context is a **packing** parameter, not just a runtime flag. At ctx 4096 the packer's
3500-token windows ended exactly at the cap for **51% of log windows and 46% of SWE
windows** — agentic tool-call chains cut mid-chain, which is the one structure this release
most needs calibrated. Repacking for 8192 (windows ≤7500) roughly doubles how much of a
chain fits in one context:

| | ctx 4096 | ctx 8192 |
|---|---|---|
| windows (same 4.4M tokens) | 1,754 | 1,136 |
| median chat window | 3,169 tok | 5,434 tok |
| **mean tool results per agentic window** | **6.1** | **13.5** |
| longest chain in one window | 26 | 42 |

Cost, measured on the 27B F16 on Metal: **48.5 s per 4096-token pass (84 tok/s) vs 116.2 s
per 8192-token pass (70 tok/s)** — about +19% wall-clock for the same token budget, since
attention is quadratic in context. A full 4.4M-token imatrix pass is ~15 h at 4096 and ~17 h
at 8192 on this machine; it is an overnight job either way, and much faster on CUDA. Trim
`--cal-logs-tokens` if that is too long.

`ctx` lives in `UniversalConfig` and the window budget derives from it, so the corpus always
records the context it was packed for (`corpora_audit.json: calibration.ctx`). **Give the
same value to all three**: `llama-imatrix -c`, `awq.calibrate(ctx=)`, `gptq.calibrate(ctx=)`.
Note 59% of agentic windows still end at the 7500 cap — these trajectories are long enough
that 16K would capture more again.

### The built dataset (2026-08-10)

Saved at `out/corpora/qwen3-universal/` (60 MB), built with the Qwen3.6 tokenizer as the
stand-in for Qwen3.8 — rebuild with the real tokenizer once the model lands, since the
calibration corpus is chat-templated and therefore tokenizer-specific. The audit is tracked
at `docs/qwen3_universal_corpora_audit.json`.

| file | what |
|---|---|
| `corpus.cal.txt` (17.4 MB) | 4,406,512 tokens · 8,650 tool calls · 5,464 tool results · 791 reasoning blocks, packed for **ctx 8192** |
| `corpus.cal.jsonl.gz` | the same, one record per window, source-labelled |
| `corpus.val.txt` | AWQ cv-scoring slice (in-domain logs + out-of-domain breadth) |
| `corpus.eval.{txt,general,tools,agentic,broad,redteam}.txt` | six disjoint eval holdouts, each needing its own `baseline.kld` |
| `sft.jsonl.gz` (22.7 MB) | 6,643 conversations · 85,643 messages · 33,572 tool calls · 4,289 reasoning turns · **24.1M tokens** |

**All three calibrators were run against this corpus, not assumed to work:**

* **imatrix** — `llama-imatrix` on the real 27B F16 (`out/exp-041/model-f16.gguf`) with
  `--parse-special`, at **both** contexts: 12 chunks @4096 and 8 chunks @8192. 496 tensor
  entries, all finite, no all-zero tensor, and the result loads through
  `calibrate.imatrix._load_base_imatrix` (the input every variant consumes).
* **AWQ** — `awq.calibrate` end-to-end on Qwen3-0.6B at ctx 4096 and **8192**: 56 groups,
  per-tensor α search with `cv_strategy="gate"` scored against `corpus.val.txt`.
* **GPTQ** — `gptq.calibrate` end-to-end at ctx 2048 and **8192** (its default is 2048, so
  8192 is a 4× jump in sequence length): 196 Hessians (112 attn / 84 mlp) accumulated from
  the strided sample, snapshotted without error.

## Stage 3 — imatrix + the four quants

```bash
PYTHONPATH=src .venv/bin/python scripts/exp060_quants_qwen38.py
```

Base imatrix at ctx 4096 → `hybrid_custom` re-weighting → IQ2_M / IQ3_M / IQ4_XS / Q5_K_M,
each with the draft layer pinned Q8_0. Each eval corpus gets its **own** FP16 baseline and
its own `results.<eval>.csv`; they are separate distributions and must never be
concatenated. `--plain-anchor` adds the un-calibrated Q2_K control; `--no-mtp-pin` exists
only to A/B what the Q8 head buys.

Equivalent single-row CLI path: `src/quant_tuner/recipes/{iq2_m,iq3_m,iq4_xs,q5_k_m}_qwen3_8_mtp.yaml`
(`quantize.mtp_pin: q8_0` resolves the layer from the GGUF the same way).

## Stage 4 — the things static metrics don't tell you

At 2 bits, KLD and agentic behavior have disagreed before (exp-051/052: AWQ lost on KLD and
won decisively on tool-call accuracy). Before believing the ladder:

```bash
PYTHONPATH=src .venv/bin/python scripts/bench_mtp_speed.py      # draft acceptance / speedup
PYTHONPATH=src .venv/bin/python scripts/run_swebench_eval.py    # agentic pass rate (Docker)
PYTHONPATH=src .venv/bin/python scripts/run_toolcall_reps.py    # tool-call accuracy, N reps
```

If IQ2_M reads badly agentically, the exp-051/052 answer was **AWQ at 2-bit** — swap the
2-bit row to `iq2_m_awq` and re-measure rather than accepting the imatrix row.

## Stage 5 — stage the repository

```bash
PYTHONPATH=src .venv/bin/python scripts/exp060_prepare_release.py
```

Stages `uploads/pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF/` with terminal-quant filenames
(so Hugging Face derives the Ollama `:IQ4_XS` tag), the calibration + eval corpora under
`calibration_data/`, an `MTP/README.md`, and a README rendered from whatever numbers exist
(missing ones render `—`). It does **not** push, and the frontmatter `license:` is emitted
as `PLACEHOLDER` — confirm the base model's terms and set it before publishing.
