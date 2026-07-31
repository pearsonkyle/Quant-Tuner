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

### Tool-call accuracy — from `quant_tuner.eval`

KLD measures token-level fidelity but doesn't tell you whether the quant still
emits *valid tool calls* on real agentic traffic. The tool-call eval — entry
point `quant_tuner.eval.run_toolcall_eval(holdout, model_path=…)` — runs each
quant against a held-out session corpus (`scripts/build_toolcall_holdout.py`
samples it with a fixed seed from `logtrain.jsonl`'s `test + holdout` slices,
fingerprint-disjoint from the calibration `train` slice). The model is served
via `llama-server`'s OpenAI-compatible endpoint (lifecycle managed by
`quant_tuner.eval.server.running_server`). Each session goes through two
independent passes:

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
type (`quant_tuner.eval.scoring.compare_value`):

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
multi-rep runner (`scripts/run_toolcall_reps.py` — see "Multi-rep eval"
below) holds one `llama-server` per model across N repetitions and emits
mean ± stdev per metric.

### MMLU-Pro reasoning — from `quant_tuner.eval.mmlu_pro`

A third axis: does the quant still solve reasoning-heavy multiple-choice
questions? Useful as a check that a quant tuned for one workload
(e.g. tool-calling on `logtrain.jsonl`) hasn't quietly regressed on
general knowledge.

Entry point `quant_tuner.eval.run_mmlu_pro_eval(holdout, model_path=…)`.
The holdout JSON (built by `scripts/build_mmlu_pro_holdout.py`) bundles
two sections:

| Section | Source | Used as |
| --- | --- | --- |
| `shots[<subject>]` | first `--n-shot` rows of the dev split | few-shot demonstrations |
| `samples` | `--n-per-subject` random rows from the test split (seeded) | the actual eval set |

Default: 25 samples × {`computer science`, `math`}, 2-shot, seed=42 — pass
`--subjects`, `--n-per-subject`, `--n-shot`, `--seed` to change any of these.

#### Prompt format

Each evaluation builds a chat-completion prompt of the form:

```
system   : You are an expert taking a multiple-choice exam. … respond with
           only the letter of the best answer.
user     : Subject: <subject>
           Question: <shot 1 question>
           Options:
           (A) … (B) … (C) … …
assistant: Answer: <truth letter>
user     : <shot 2 question, same format>
assistant: Answer: <truth letter>
user     : <target question>      ← model responds here
```

The number of demonstration pairs equals `holdout.n_shot`. MMLU-Pro has up
to 10 answer choices per question (A–J), so the parser bounds the valid
letter range by `len(options)` — a 4-option question rejects an "F"
prediction even if the model emits one.

#### Answer extraction

`parse_answer` walks four tiers in priority order:

1. Explicit marker: `Answer: X` (or `Answer = X`, `Answer - X`, `Answer:(X)`,
   case-insensitive)
2. Parenthesized: `(X)` anywhere in the completion
3. Verb phrase: `answer is X` / `the answer is (X)`
4. Fallback: first standalone capital letter A–J in the completion

A reply of "B" alone, "Answer: (B)", "The answer is B", or "After working
through this, B is right." all resolve to `"B"`. A reply with no in-range
letter at all (or a letter past the option count) is logged as
`pred=None`, counted as `n_unparseable`, and treated as a wrong answer for
accuracy purposes.

#### Output

`run_mmlu_pro_eval` returns an `MmluProSummary` dataclass with:

* `accuracy` — overall, fraction of correct predictions
* `by_subject[<subject>]` — `{n, correct, accuracy, unparseable}` per subject
* `n_unparseable` — completions where no letter could be extracted
* `per_sample` — full row list for debugging (also written as JSONL log)

The CLI shim `scripts/eval_mmlu_pro.py` appends one CSV row per
`(model, run)` with per-subject accuracy as broken-out columns.

## Multi-rep eval (`quant_tuner.eval.reps`)

Any benchmark whose summary reduces to `dict[str, float]` can plug into the
generic multi-rep runner — `eval.reps.run_reps_for_models` spawns one
`llama-server` per model, runs the eval `N` times against it with a per-rep
seed (`base_seed + rep_idx`), and aggregates **mean ± stdev** across reps.

Defaults to **10 reps**; override with `--reps` on the CLI or `reps=N` in
Python. Set it to `1` for a quick smoke test, `25+` for tighter confidence
intervals.

```bash
# 10 reps × 8 GGUFs (~14 h at default settings)
uv run python scripts/run_toolcall_reps.py

# Just two models, 5 reps
uv run python scripts/run_toolcall_reps.py \
    --models out/run/model-f16.gguf out/run/Q4_K_M-none.gguf \
    --reps 5

# MMLU-Pro, same shape
uv run python scripts/run_mmlu_pro_reps.py --models out/run/*.gguf --reps 10

# Deterministic (no stdev → 1 rep is enough)
uv run python scripts/run_mmlu_pro_reps.py --temperature 0 --reps 1
```

Both runners emit **two CSVs**:

| File | Shape | Use |
| --- | --- | --- |
| `<name>_reps_results.csv` | one row per `(model, rep)` | debugging individual reps |
| `<name>_reps_aggregated.csv` | one row per model | leaderboard input |

The aggregated CSV expands each metric into `<m>_mean` / `<m>_stdev` /
`<m>_n` columns, plus the sampling params (`temperature`, `top_p`, …) for
traceability. The leaderboard aggregator (`leaderboard.aggregate.merge_toolcall`)
detects this shape automatically and renders cells as `mean ± stdev`.

To plug a new benchmark into this pipeline, write a `dict[str, float]`-returning
adapter and call `run_reps_for_models`:

```python
from quant_tuner.eval.reps import run_reps_for_models, write_csvs, sampling_extra_cols
from quant_tuner.eval.toolcall import Sampling

def my_eval(base_url: str, sampling: Sampling, rep: int) -> dict[str, float]:
    # … run one rep against the model at base_url …
    return {"my_metric_1": 0.85, "my_metric_2": 0.42}

sampling = Sampling(temperature=0.6, top_p=0.95, top_k=20)
by_model = run_reps_for_models(
    models=[Path("model.gguf")],
    eval_fn=my_eval,
    reps=10,
    sampling=sampling,
)
write_csvs(by_model,
           per_rep=Path("out/my_reps.csv"),
           aggregated=Path("out/my_agg.csv"),
           extra_cols=sampling_extra_cols(sampling))
```

`run_reps_for_models` handles server lifecycle (spawn/health-check/teardown
per model), error capture (a failed rep gets `error=…` and is skipped during
aggregation), and per-rep / per-model callbacks for progress logging.

## Red team (safety)

Every other metric in this document answers *"is the quant still capable?"* —
BPW, KLD, perplexity, tok/s, tool-call accuracy, MMLU-Pro, SWE-rebench pass rate.
None answers *"is the quant still safe?"* A quant that had lost its refusal
behaviour entirely would score clean on every column the leaderboard renders.

### Why this exists

Anthropic's [position on open-weights models][pos] (Dario Amodei, 27 Jul 2026)
proposes that "all sufficiently capable models, open and closed, should go
through mandatory safety testing" — testing "for cyber, biological, and alignment
risks **before release**" — and is explicit that the open-weights risk question
is an empirical one:

> Whether open models do or don't pose an increased risk, and whether that risk
> can be mitigated, is something that should emerge from testing, rather than be
> decided in advance […] Questions like this should be empirically answered by
> rigorous pre-release testing, not assumed in advance.

The gap this repo sits in is **"before release."** Every mitigation the piece
proposes gates the artifact at the moment of publication. A GGUF quantization is
produced *after* release, by a third party, from the released weights — it is not
the artifact that was tested, and nothing re-tests it.

The piece's model of what happens post-release is deliberate removal. In Amodei's
own words, open-weights models "do potentially present a higher risk than closed
models, because it is very difficult to apply guardrails to them or monitor their
usage, and once weights are released they cannot be withdrawn." Its footnote 2
quotes the **UK AI Security Institute** (this is the report's language, not
Amodei's) — "safeguards can be removed, and copies can be downloaded,
redistributed, and run on private systems beyond monitoring."

*Removed* — an act. Neither has a category for safeguards that **degrade
incidentally**, as a side effect of a routine engineering step performed by
someone with no interest in removing them. That is what quantization is. The
piece discusses distillation at length, but purely as a compute-evasion and IP
problem; whether a *derived* artifact inherits its ancestor's safety properties
is never raised, and quantization, LoRA and merges are not mentioned at all.

It also points at tamper-resistance work ("promising methods for improving the
safety of open-weights models, including recent research from AE Studio and
Anthropic on modular training strategies"), which sharpens the same point: any
claim that a safety property is *durably* baked into open weights needs a test
that the property **survives derivation**.

[pos]: https://www.anthropic.com/news/position-open-weights-models

### How it works

Three independent OpenAI-compatible endpoints, by design:

| Role | What it is | Why separate |
| --- | --- | --- |
| **target** | the quant under test (`llama-server` on a local GGUF) | the thing being measured |
| **simulator** | writes the attack prompts | a safety-tuned model refuses to author attacks |
| **judge** | grades each response 1 (defended) / 0 (complied) | a 2-bit quant judging its own jailbreaks produces garbage |

`eval/red_team.py` wraps each endpoint in a `DeepEvalBaseLLM` (`LocalLLM`) and
hands them to [`deepteam`](https://github.com/confident-ai/deepteam). Vulnerability
and attack selection is plain YAML (`eval/red_team_configs/`), resolved by name
through `_VULN_SPECS` / `_ATTACK_SPECS`.

Three adaptations that matter for self-hosted models:

1. **`schema=` raises `TypeError`** so deepteam falls back to its raw-text
   `trimAndLoadJson` parser — GGUF endpoints don't implement deepeval's
   structured-output path. The prompt is remembered, so deepteam's schema-free
   retry of that same prompt is sent with llama.cpp's
   `response_format={"type":"json_object"}`: JSON enforcement exactly where JSON
   is wanted, and nowhere else (some simulator calls legitimately return prose).
2. **`<think>` / `<thinking>` blocks are stripped** before any parsing — reasoning
   traces contain braces that break the JSON parser.
3. **The target callback takes `(input, turns=None)`.** deepteam's
   `wrap_model_callback` forwards conversation history *only* to a callback
   declaring more than one parameter. With one parameter, every multi-turn
   jailbreak (Linear / Crescendo / Tree) silently probes a target with no memory
   of the escalation, and those scores mean nothing. `tests/unit/test_red_team.py`
   asserts the arity.

### The frozen attack bank

deepteam simulates a fresh attack bank per run. Two quants scored independently
therefore differ both by the model *and* by the prompts they happened to be
asked — and at realistic bank sizes the second effect swamps the first.

`--frozen-bank` (default on for multi-target runs) simulates the bank **once**,
against the first target, and replays it verbatim against every later target.
Pass the F16 reference first so the prompts are written against the unquantized
parent. `--bank-out` dumps it; `--bank-in` replays a bank from weeks earlier, so
a new leaderboard row stays comparable to old ones.

Errored cases stay in `n_tests` but are excluded from `pass_rate`'s denominator,
and `score` stays **tri-state** (1 / 0 / `None`) all the way through the CSV. A
timeout is not a jailbreak. `run_red_team_eval` also **raises** when zero cases
scored — deepteam defaults `ignore_errors=True`, so an unreachable target would
otherwise report a clean `pass_rate=0.0`, indistinguishable from a model that
complied with everything (same guard, same reasoning, as `toolcall.py`).

### The ladder — the number worth reading

`scripts/redteam_ladder.py` joins two runs on a content-derived `case_id`
(`sha1(vulnerability|type|attack|input)`) and reports a McNemar-style paired
breakdown rather than a bare pass rate:

| Column | Meaning |
| --- | --- |
| `n_flip_unsafe` | reference refused, quant complied — **incidental safeguard degradation** |
| `n_flip_safe` | the reverse; the noise floor / over-refusal |
| `net_drift` | `(n_flip_safe − n_flip_unsafe) / n_paired` |
| `pass_rate_delta` | quant − reference, over paired cases only |
| `mcnemar_p` | exact two-sided p for the flip asymmetry |
| `n_unmatched` | **non-zero means the bank was not shared — the row is invalid** |

`n_flip_unsafe` is the quantity the position piece's framing has no slot for: no
adversary, no fine-tuning, a routine engineering step, and the artifact that
ships is not the artifact that was tested. Cases errored on *either* side are
dropped, since they say nothing about either model. `unsafe_flips.csv` writes out
the actual evidence, so a number can be eyeballed before it is cited.

With `--reps > 1`, a case counts as defended only if the model refused on **every**
rep. That is the conservative direction: a model that complies one time in three
has not reliably refused.

### Does any existing gate see it?

`scripts/redteam_vs_quality.py` correlates the ladder's drift against `mean_kld`,
`ppl_ratio`, `same_top_p`, tool-call accuracy and MMLU-Pro (Spearman, so the
question is monotonic over a handful of rungs). Pure CSV analysis — no model, no
GPU.

The expected answer is **no correlation**, and that is the useful result: refusal
is low-probability-mass behaviour that an averaged divergence washes out, and the
pipeline's own guardrails (`ppl_max_ratio`, `sanity_max_rel` in `calibrate/gptq.py`)
are deliberately *relaxed* at 2-3 bits — exactly where alignment is most fragile.
If it holds, "we validated the quant" as currently practised says nothing about
whether the safety properties survived. That is a measured claim rather than an
assumed one, which is the standard the position piece asks for.

### Agentic (`red_team_agentic.yaml`)

The position piece names **cyber** as a test axis, and this repo's models are
tool-calling coding agents — yet a chat-only probe grades prose, not action.
`scripts/redteam_agentic.py` + `eval/red_team_agent.AgenticTarget` give the target
a real `bash` tool inside a disposable SWE-rebench container (the same mechanism
`eval/swebench.py` already uses to run untrusted model output — nothing executes
on the host). The returned `RTTurn` carries `tools_called`, which deepteam's
agentic metrics grade against, so "complied" means *ran the command*.

`seed_file()` plants the attack in the checkout itself, which is the realistic
vector for `IndirectInstruction` / `document_embedded_instructions`: a poisoned
README or test fixture, not a user turn.

> ⚠️ **Capability confound.** Complying with an agentic attack requires
> successfully emitting a tool call. A quant too degraded to tool-call scores as
> "safe" for entirely the wrong reason. The runner prints `n_tool_calls` beside
> `pass_rate` and warns when a target executed nothing; a rung whose pass rate
> rises while its tool-call count collapses has not become safer. Commands
> written in prose (```` ```bash ````) are counted too, precisely so the most
> degraded rungs don't win by incompetence.

### Scope — what this does not measure

* **Bio.** The piece names cyber, biological, and alignment. A bio-uplift probe
  set does not belong in a public quantization repo; that axis is out of scope by
  choice, not oversight.
* **Absolute safety.** These are *relative* measurements against an ancestor. A
  high `pass_rate` means "no worse than F16 on this bank", not "safe".
* **Bank coverage.** A frozen bank is a sample. `n_flip_unsafe = 0` means this
  bank found nothing, not that nothing is there.
* **Judge quality.** The judge is itself a local model doing prompt-and-hope JSON.
  Check `n_errored` before trusting any rate.

### Roadmap

Two scenarios designed but not yet run, both aimed at the same gap:

* **Corpus provenance.** quant-tuner calibrates on a *user-supplied* prompt/response
  log, and (since the QAT work landed) also fine-tunes on self-generated solver
  trajectories — the AWQ α-search, the GPTQ Hessian, and `qat/train.py`'s masked CE
  all optimise against that corpus with no refusal-preservation term. Score two
  otherwise-identical artifacts that differ only in corpus. If it moves, corpus
  choice is an alignment-relevant *invisible* knob: the output is a standard GGUF
  with no runtime cost, and nothing in the artifact records which corpus produced
  it — cheaper and more deniable than the fine-tuning the discourse assumes. It
  would argue for a corpus fingerprint in the GGUF kv (precedent: the
  `quant_tuner.lens.*` provenance kv in `lens/gguf_edit.py`).
* **Refusal-direction survival.** Reuse `lens/probes.py` with a refusal probe set
  to ask not "does it refuse less" but "did the 2-bit grid destroy the
  representation that drives refusal" — the mechanistic complement to the ladder,
  and the shape of test any tamper-resistance claim needs.

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
