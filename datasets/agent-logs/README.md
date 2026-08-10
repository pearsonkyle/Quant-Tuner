# agent-logs — the local calibration log corpora

**Not a publishable dataset, and not versioned.** Unlike its siblings in `datasets/`, this
directory is not in `datasets/registry.REGISTRY` and has no manifest or push path: the CLI
logs are real captured usage (prompts, file contents, paths), this repo is public, and they
are not ours to publish. It lives here because it *is* a dataset the calibration pipeline
consumes, and putting it under `datasets/` means the payloads fall inside the existing
`datasets/**/data/` gitignore rather than sitting loose at the repo root.

**Only this card is tracked. `data/` is local-only** — nothing recreates it from the repo, so
back it up separately. Everything below describes what has to be in `data/` for the corpus
builders to run.

Both files are gzipped and read transparently — see `quant_tuner.data.ingest`, which sniffs
the row format so every consumer sees one session schema.

| File | Rows | What it is |
|---|---|---|
| `data/logs-cli.jsonl.gz` | 253 sessions | **CLI usage logs.** Interactive coding sessions captured from Claude Code / opencode / qwen code, scraped with [LogMiner](https://github.com/pearsonkyle/LogMiner). Previously `logtrain.jsonl` at the repo root (58 MB → 11.8 MB gzipped; content verified byte-identical through the move). |
| `data/logs-agents.jsonl.gz` | 435 trajectories | **Harvested agent trajectories.** Verified (tests-passed) issue-solving runs, one row per run. |

## Row schemas

`logs-cli.jsonl.gz` — the original log-export shape:

```
id, source ("claude" | "qwen" | …), score, metrics {tool_calls, …},
messages: [JSON-encoded string, …]        # note: strings, parsed by ingest.normalize_messages
```

`logs-agents.jsonl.gz` — the harvested-trajectory shape:

```
messages: [{role, content, tool_calls?}, …]   # dicts, with STRUCTURED tool_calls
tools:    [{type: function, function: {...}}] # the schemas the agent was actually given
meta:     {model, agent, language, instance_id, repo, resolved, grade_method,
           n_fail_to_pass, n_pass_to_pass, tools_used, total_tokens, wall_sec,
           has_reasoning, n_tokens}
```

`ingest.load_sessions` normalizes the second shape into the first's session schema:
`score = 1.0` iff `meta.resolved`, `metrics.tool_calls` counted from the messages, and
`source = "agents:<language>"`.

## Coverage of the agent trajectories

435 rows over **94 unique issues**, all `resolved=true`:

* **19 languages** — python 75, rust 55, php 52, c 48, ts 37, go 33, cpp 25, java 23, js 22,
  csharp 16, swift 16, elixir 10, clojure 8, r 6, lua 4, kotlin 2, dart 1, ocaml 1, scala 1.
* **7 agent scaffolds** — agentloop 100, claude-code 71, codex 57, gaiaa-agents 50,
  gaiaa-agents-codegraph 55, mini-swe 53, openai-agents 49.
* **3 solver models** — claude-4-8-opus 265, claude-sonnet-5 114, laguna-m1-fp4 56.
* 18,051 assistant turns carrying tool calls; 18,852 tool results.

## Two things that matter when using these

**Split by issue, not by row.** ~4.6 rows share each `instance_id` (the same issue solved by
different scaffolds/models). `split.split_sessions` groups by `ingest.session_group`, so every
attempt at an issue stays on one side of the train/test/holdout split — otherwise the eval
holdout would contain another attempt at an issue that calibration already saw, and would be
measuring fit rather than generalization.

**No overlap with the agentic eval.** Checked at import time of this corpus: the 94 issues are
disjoint from both `datasets/swe-agentic-trajectories` (278 ids) and the SWE-rebench eval
holdout (`out/external/swe-rebench/holdout.jsonl`). Re-check that if either is regenerated.

## Where they are consumed

* `quant_tuner.data.universal` / `scripts/build_universal_corpus.py` — **both** files, as the
  `logs` source of the four-source calibration corpus; the holdout slice becomes
  `corpus.eval.tools.txt`.
* `scripts/build_corpora.py` — the CLI logs **only**, deliberately: it reproduces the
  published two-source runs, whose numbers were produced before the agent logs existed.
* `quant_tuner.qat.corpus`, `scripts/build_toolcall_holdout.py` — the CLI logs.
