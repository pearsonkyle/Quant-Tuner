# Benchmarks

`quant-tuner` evaluates a quantized GGUF along three independent axes:
**fidelity** (does it produce the same outputs as the F16 reference?), **size**
(how compact is it on disk?), and **speed** (how fast is inference?).

The bench runner (`quant_tuner.bench.runner.bench_one`) produces one CSV row
per model with all of the metrics below.

## Metrics

### Fidelity — from `llama-perplexity --kl-divergence`

These compare the quantized model against an F16 reference on a held-out
text corpus. The reference is captured once via
`kld.build_baseline(reference_gguf, eval_dataset, baseline.kld)`; every
subsequent evaluation reuses that baseline.

| Column          | What it measures                                                                       |
| --------------- | -------------------------------------------------------------------------------------- |
| `ppl`           | Mean per-token perplexity of the **quantized** model on the eval set. Lower is better. |
| `ppl_ratio`     | `ppl(Q) / ppl(F16)`. 1.000 means quant matches reference exactly. Closer to 1 is better. |
| `mean_kld`      | Mean KL-divergence per token between Q's and F16's next-token distributions. Lower is better. |
| `median_kld`    | Median per-token KLD. Outlier-robust; dominates the mean when a quant has a few catastrophically off tokens. |
| `same_top_p`    | Percentage of tokens where Q and F16 agree on the top-1 prediction. Closer to 100% is better. |
| `rms_dp`        | RMS of the per-token difference in top-1 probability. Lower is better.                 |

KLD is the most informative of these for ranking quants — PPL ratio is
coarse-grained because PPL itself is an exponential of cross-entropy, so
small differences hide large ones. Mean KLD splits hairs that PPL ratio
can't.

### Size — from GGUF on-disk inspection

| Column     | Source                                              |
| ---------- | --------------------------------------------------- |
| `size_gib` | Filesystem byte size / 1024³                        |
| `bpw`      | `size_bytes · 8 / n_params` — effective bits/weight |

`n_params` is computed once per F16 reference (`bench.bpw.n_params(f16_gguf)`)
and reused; the quantized files don't store a separate count.

### Speed — from `llama-bench -o json`

| Column           | What it measures                                              |
| ---------------- | ------------------------------------------------------------- |
| `prefill_tok_s`  | Avg tok/s during a 2048-token prompt eval (no generation).    |
| `decode_tok_s`   | Avg tok/s during 128-token decode (no prompt).                |
| `ttft_2k_ms`     | Derived: `2048 / prefill_tok_s · 1000`. Time-to-first-token for a 2k-prompt request. |

Decode speed is the one users feel during a chat session; prefill matters for
agent workflows that send long contexts. Heavy K-quants (Q5_K_M, Q6_K) tend
to have lower decode tok/s than the F16 baseline on CPU/Metal because the
dequantize cost per matmul exceeds the I/O savings.

### Tool-call accuracy — from `scripts/eval_toolcall.py`

KLD measures token-level fidelity but doesn't tell you whether the quant still
emits *valid tool calls* on real agentic traffic. The tool-call eval runs each
quant against a held-out session corpus
(`scripts/build_toolcall_holdout.py` — sampled with a fixed seed from
`logtrain.jsonl`'s `test + holdout` slices, fingerprint-disjoint from the
calibration `train` slice). The model is served via `llama-server`'s
OpenAI-compatible endpoint; the script does two independent passes per
session:

**1. Per-turn pass.** For every ground-truth assistant turn that emitted a
`tool_calls`, replay the prior context and ask the model for *one* completion.
Compare the model's first tool call to the recorded one. By default the
session stops at the first selection failure (`--no-stop-on-fail` keeps
scoring further turns).

**2. Rollout pass.** Once per session, run the full agentic loop from the
first user turn. When the model emits a tool call, splice in the *recorded*
tool result (matched by tool name, in call order) and continue. Capture
whether the model reaches a natural stop and which tools it used.

These produce four columns in the leaderboard (`out/<run>/toolcall_results.csv`,
merged into `LEADERBOARD.md`):

| Column        | Pass     | What it measures                                                                                |
| ------------- | -------- | ----------------------------------------------------------------------------------------------- |
| `Tool Sel %`  | per-turn | `tool_selection_acc` — fraction of turns where `pred.name == truth.name` (case-insensitive).    |
| `Param Acc %` | per-turn | `param_acc_mean` — mean across turns of `(required-arg hits) / (required args)`, type-aware.    |
| `Schema %`    | per-turn | `schema_valid_rate` — fraction of predicted calls that pass JSON-schema validation against the session's declared tools. |
| `Rollout %`   | rollout  | `rollout_complete_rate` — fraction of sessions where the model reached a natural stop within `--rollout-max-turns`.       |

A fifth metric, `rollout_tool_set_match_rate` (predicted tool *set* equals the
recorded `tools_used` set), is logged to CSV but not currently displayed.

#### Per-arg scoring (`param_acc`)

A predicted call's parameter score is `(hits) / (required keys)`, where each
key counts as a hit if it's present **and** its value is *equivalent* to the
ground truth. Equivalence is type-aware, dispatched by key name and JSON-schema
type (`eval_toolcall.compare_value`):

| Argument shape           | How values compare                                                            |
| ------------------------ | ----------------------------------------------------------------------------- |
| boolean                  | strict equality                                                               |
| number / integer         | exact, or relative tolerance ≤ 10 % (`max(\|a\|, \|b\|, 1)` denominator)      |
| filesystem path keys     | `os.path.normpath` equality; case-insensitive or same-basename → "similar"    |
| shell-command keys       | trimmed-equality, or shlex tokens with same `argv[0]` and ≥ 30 % Jaccard      |
| list / dict (structural) | canonical-JSON equality                                                       |
| free-text keys[^1]       | presence-only (any non-null counts)                                           |
| generic strings          | Jaccard over word tokens ≥ 0.5                                                |

[^1]: `command, description, content, message, text, prompt, question(s), new_string, old_string, code, body, instructions, explanation, summary, thought, thinking, query, search` — keys where two competent agents will rarely emit byte-identical strings.

"Similar" matches count as hits; only outright mismatches dock the score.
This was deliberate: with greedy decoding the model can pick the right tool
and right path but phrase a `description` differently, and we don't want to
penalize that as a parameter error.

#### Schema validation (`schema_valid`)

For each predicted call, the script looks up the JSON-schema declared by the
session's `tools` list (per OpenAI's function-calling format) and validates
`arguments` against it. A call fails schema if (a) the tool name isn't in
the tools list, (b) `arguments` doesn't parse as a JSON object, or (c)
`Draft7Validator` reports errors. When the `jsonschema` library is missing,
a fallback checks only `required` keys (logged as `"ok (no jsonschema lib)"`).

Note: `tool_selection_acc` and `schema_valid_rate` are *independent* —
a wrong-tool call can still pass schema if the tool name happens to exist
in the tools list and its args validate, and a right-tool call can fail
schema if `arguments` is malformed.

#### Sampling

Defaults to greedy (`temperature=0`). The runner forwards OpenAI-standard
params (`top_p`, `presence_penalty`, `seed`) directly and llama.cpp
extensions (`top_k`, `min_p`, `repeat_penalty`) via `extra_body`. The
multi-rep runner (`scripts/run_toolcall_reps.py`) holds one `llama-server`
per model across N repetitions and emits mean ± stdev per metric.

## Bench suites

`bench_one(quant, label, ..., suite=...)` selects what to compute:

| Suite          | Computes                       | Needs              |
| -------------- | ------------------------------ | ------------------ |
| `quick`        | size + BPW only                | nothing            |
| `kld`          | size + BPW + KLD               | eval_dataset, eval_baseline |
| `speed`        | size + BPW + llama-bench       | nothing            |
| `full`         | all of the above               | eval_dataset, eval_baseline |
| `leaderboard`  | alias for `full`               | "                  |

A `quick` row in CI/dev is enough to catch obvious regressions before paying
for the full KLD pass (which runs llama-perplexity on the full eval set and
takes minutes on a 1B model, tens of minutes on a 9B).

## Train / test / holdout splits

`quant_tuner.data.split.split_sessions(...)` partitions log sessions by
fingerprint into three disjoint sets:

* **train** — used by the calibrators (imatrix forward pass, AWQ activation
  capture, GPTQ Hessian accumulation). Default 80%.
* **test** — used to build the KLD eval dataset that scores every quant.
  Default 10%.
* **holdout** — reserved for end-to-end checks like the tool-call benchmark
  or human eval. Never touched by calibration or KLD scoring. Default 10%.

The split is deterministic given a `seed`. Splits are session-level (not
token-level), so a single session never appears in more than one split — this
prevents contamination from large multi-turn sessions whose tokens would
otherwise leak across the train/eval boundary.

## Comparing across models

KLD is computed against a *specific* reference GGUF (typically F16 of the
same model). To compare `quant-tuner` results across two different base
models, you'd need either:

1. A common eval set + a common reference (e.g. both models scored against
   a third "judge" model's distribution), or
2. Comparing only `same_top_p` / `rms_dp` numerically, since those depend
   only on the relative shape of each model's output distribution.

In practice for tuning one model, sticking with the per-model F16 reference
is what you want — every row's KLD is "how much fidelity did this quant
lose relative to my own F16?", which is the directly-actionable number.

## SQS — the leaderboard summary scalar

(Coming in Phase 5.) The leaderboard collapses all of the above into one
sortable number `SQS` (Squashed Quality Score) using weights `α, β, γ` over
KLD, top-1 agreement, and speed retention vs F16. Until the aggregator
lands, treat `mean_kld` as the primary ranking column and `decode_tok_s` as
the speed tie-breaker.
