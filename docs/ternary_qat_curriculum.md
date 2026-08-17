# The three-round ternary-QAT curriculum

A staged continued-QAT run for `prism-ml/Ternary-Bonsai-8B`, replacing the single-corpus
runs (`sft8k-full`, `sft32k`, `sft32k_sw1`) with an ordered curriculum. Companion to
`docs/ternary_qat.md` (method), `docs/ternary_qat_sft32k_study.md` (the observations that
set the hyper-parameters) and `docs/qat_32k_handoff.md` §10 (the CUDA port).

## The three rounds

| round | corpus | what it teaches | rows | tool calls | reasoning |
|---|---|---|---|---|---|
| 1 | `HuggingFaceH4/ultrachat_200k` | broad conversational grounding | 207,865 | 0 | 0 |
| 2 | `r0b0tlab/qwen3.8-max-…-distillation` | tools, agents, reasoning | 17,216 | 27,672 | 34,942 |
| 3 | our universal SFT | CLI logs + trajectories that resolve real issues | ~3,000 | — | — |

Each round **continues from the previous round's latents**, so this is one long fine-tune
over a changing distribution, not three independent runs. The ordering is
general → capability → in-domain, so the last thing the model sees is the distribution it
is graded on. Catastrophic forgetting runs the other way, and that is the point.

Build and run:

```bash
python scripts/build_external_sft.py ultrachat --out out/corpora/round1-ultrachat
python scripts/build_external_sft.py distill   --out out/corpora/round2-distill
bash scripts/run_curriculum_qat.sh curriculum 5e-4 1.0
```

Round 3's corpus is the existing `scripts/build_universal_corpus.py` output; nothing new
is needed for it.

## Why each round is budgeted, not epoch-capped

ultrachat is ~300M tokens. At a 32768 window that is ~5,500 steps — **~96 h on its own**,
against the ~11 h the whole `sft32k` run took. Rounds are therefore capped by token budget
(`BUDGET_R1=ultrachat=20000000`), sized so each round is a comparable slice of wall-clock
and the full curriculum lands near ~33 h. Rounds 2 and 3 are already ~14M and ~20M tokens
and are taken whole.

Budgets are per-source `SOURCE=TOKENS` pairs — there is no global cap, because
`build_sft_qat_corpus.py` budgets each source separately so one source cannot crowd out
another.

## What is settled, and what this curriculum does not decide

Carried forward from the sft32k study, not re-litigated here:

* **fp32, never `--compute-dtype bf16`.** bf16 is 5.15× faster and diverged twice,
  non-reproducibly, at different steps, with every corpus source spiking at once and no
  anomalous gradient beforehand (Obs. 8). `GradSpikeGuard` cannot catch it: the weights are
  already broken by an update whose norm was *below* threshold, and the huge norms that
  follow are a symptom. `run_curriculum_qat.sh` does not expose the flag.
* **`--matmul-precision high` (TF32) IS safe and is the default here.** A different knob:
  latents, the TWN threshold and `ternarize_group` are elementwise fp32 and stay bit-exact,
  so the codes a step produces are unperturbed. Only matmul accumulation is reduced, and
  TF32 keeps 10 mantissa bits against bf16's 8. Worth ~1.38× at this window.
* **Read the code flips, not the loss.** A ternary model only learns by flipping codes; at
  lr 3e-4 the loss falls on scale drift alone with ~0% flipped.
* **gnorm magnitude is not the failure signal.** 90.96 at step 55 was survived without
  incident while 15.80 at step 265 diverged. NaN/Inf is the unambiguous one.

Still open, and this curriculum does **not** answer them:

* the lr 2.5e-4 control, which would separate "bf16 is causal" from "a 32K window at
  lr 5e-4 is simply fragile";
* whether the stop weight needs to be >1.0 — that is what `sft32k_sw1` measures, and its
  answer should set `STOP_WEIGHT` here before round 1 starts.

## The four dataset traps

Encoded in `data/external_sft.py` and unit-tested in `tests/unit/test_external_sft.py`.
Each one produces a corpus that builds, trains and exports without error while teaching the
model something wrong.

**1. `sft_tools` and `sft_agent` are the same rows.** Byte-identical parquet (4,226,831
bytes) and an identical id-set hash over all 5,337 rows. The Hub's `/size` API reports every
one of the 24 configs with the same row count and byte size, which makes them look distinct
— they are not. Taking both double-weights the agentic half of round 2. `DISTILL_ALIASES`
records it; the builder skips the alias and says so.

**2. Tool schemas nest `parameters_json` as a JSON *string*.** The Qwen3 chat template
renders tools with `tool | tojson`, so an unconverted row emits the whole parameter schema
as one escaped blob inside `<tools>`. It templates cleanly, tokenizes cleanly, and teaches
a doubly-encoded format that no inference-time caller produces. `normalize_tool` converts
it; verified by rendering a raw row and confirming `parameters_json` leaks verbatim without
it.

**3. The per-row `split` field is `"train"` in every file**, including the validation and
test parquet. Trusting it puts graded rows into training. The *file* the rows came from is
the real split, so `convert_distill_rows` takes it as an argument: `UPSTREAM_SPLIT_MAP`
sends their `validation` → our `test` (the trainer's val corpus) and their `test` → our
`holdout`.

**4. Reasoning is inline `<think>` in content, never `reasoning_content`** — despite that
field existing on every message. `data.reasoning` normalizes both forms, so this works, but
*counting* it needs the non-empty check: the Qwen3 template emits a bare `<think></think>`
on the final assistant turn when no reasoning is supplied, which reported **200 reasoning
blocks on ultrachat, a corpus with exactly zero**.

## `sft_science` and eval contamination

`sft_science` is 10,189 rows, and 9,426 of them come straight from public benchmark sets:
SciQ (3,541), CommonsenseQA (2,528), QASC (1,533), ARC-Easy (871), OpenBookQA (527),
ARC-Challenge (426). Only `k3_science_logic_data` (763) is synthetic.

Training on this contaminates any multiple-choice eval — MMLU-Pro overlaps the same
material directly, and `scripts/run_mmlu_pro_reps.py` is part of this repo's leaderboard.
`--drop-benchmarks` removes them, leaving 763 rows. Two defensible positions:

* include it and **stop quoting MMLU-Pro for this model**, or
* `--drop-benchmarks` and keep the eval meaningful.

It is also the round-2 source least aligned with the goal: single-turn multiple-choice
answering is not the agentic distribution we are trying to reach, and it is 2/3 of round 2
by row count. `sft_tools` is where the value is — 100% tool coverage, median 11 turns,
27,672 tool calls.

## Measured compatibility

Round 2 through the real builder (`--window 8064 --max-tool-tokens 3072 --min-density
0.05`):

```
distill-science  10189 convs   4,714,391 tok  63% masked -> 584 windows
distill-tools     5337 convs   9,671,000 tok  51% masked -> 1199 windows
TOTAL 14,385,391 tokens (55% assistant-masked) -> 1783 windows of 8064
tool-calls 35661/24987 · reasoning 39579/40935 kept (96.7%)
labeled <|im_end|> targets: 34613
window trainable-density deciles: 0.35 0.43 0.45 0.48 0.51 0.55 0.59 0.62 0.63 0.65 0.94
```

Two things to note. Reasoning survives at 96.7% because these are agentic trajectories —
the template keeps reasoning only on assistant turns *after the last user turn*, which is
nearly the whole conversation when there is one task turn followed by many tool turns.
And the density deciles are far above the log corpus's, so `--min-density 0.05` drops
nothing here.
