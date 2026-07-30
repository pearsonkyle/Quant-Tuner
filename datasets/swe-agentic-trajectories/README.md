---
license: other
task_categories:
- text-generation
tags:
- agentic
- swe-bench
- code
- tool-use
- distillation
- trajectories
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: resolved
    path: data/resolved.jsonl
  - split: all
    path: data/all.jsonl
---

# SWE Agentic Trajectories (verified)

Multi-step agentic software-engineering trajectories on real GitHub issues, graded by running the hidden tests. Chat-template ready.

**Version `0.1.0`** · built 2026-07-30T11:38:05

| split | rows | verified (tests pass) | mean tool calls | size |
| --- | ---: | ---: | ---: | ---: |
| `resolved` | 71 | 71 | 34.3 | 4.0 MB |
| `all` | 278 | 71 | 39.2 | 19.4 MB |

## What one row is

One row is a **complete agentic session for a single real GitHub issue** — not a
question/answer pair. The agent reads the issue, then over many steps runs shell commands,
reads and greps files, edits source, runs the project's tests, reads failures, edits again,
and finally submits a patch. Each step is a tool call plus the output it received. Sessions
average ~39 tool calls and tens of thousands of tokens.

## Provenance

* **Issues**: [nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) — real
  GitHub issues with hidden `FAIL_TO_PASS` / `PASS_TO_PASS` test sets.
* **Solver**: [Ornith-1.0-9B](https://huggingface.co/pearsonkyle/Ornith-1.0-9B) (Q5_K_M),
  driven by the OpenAI Agents SDK against a local `llama-server`, one clean Docker container
  per instance (the SWE-rebench image), `temperature=0.25`, max 100 steps.
* **Grading**: every trajectory was graded by actually running the gold tests in the container.
  `resolved=true` means the hidden tests **passed** — these are verified solutions, not merely
  plausible-looking diffs.

## Splits

* `resolved` — only trajectories whose hidden tests passed. Use this for distillation /
  behavior cloning where you want outcome-correct supervision.
* `all` — every graded trajectory, including failed and empty-patch attempts. Useful for
  preference/critic data or failure analysis. Check the `resolved` field per row.

## Using it

Records are already in chat format, so they render with any chat template:

```python
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("<your-model>")
rec = json.loads(open("data/resolved.jsonl").readline())
text = tok.apply_chat_template(rec["messages"], tools=rec["tools"], tokenize=False)
```

`messages` uses the standard roles (`system`, `user`, `assistant` with `tool_calls`, `tool`),
and `tools` carries the single `bash(command)` schema the agent was given, so schema-conditioned
tool use trains correctly.

## Caveats

* The solver is a 9B model: trajectories are competent but not expert, and often take
  exploratory detours before landing the fix.
* `all` contains failures by design. Filter on `resolved` unless you want them.
* Tool outputs are raw container stdout/stderr and can be long; truncate as needed.
* **Licensing**: the issue text and repository content originate from the upstream GitHub
  projects via SWE-rebench and retain their own licenses. Treat this as a derived research
  artifact and check upstream terms before commercial use.

## Row schema

| field | meaning |
| --- | --- |
| `instance_id`, `repo` | the upstream issue and its repository |
| `messages` | the full session in chat format (`system`/`user`/`assistant`+`tool_calls`/`tool`) |
| `tools` | tool schema the agent was given (`bash(command)`) |
| `submission` | the final `git diff` the agent produced |
| `resolved` | **true = the hidden tests passed** (verified solution) |
| `patch_produced`, `patch_chars` | whether a non-empty patch was submitted, and its size |
| `n_messages`, `n_tool_calls`, `tools_used`, `tool_errors` | session shape |
| `n_fail_to_pass[_passed]`, `n_pass_to_pass[_passed]` | grading detail |
| `prompt_tokens`, `completion_tokens`, `total_tokens`, `wall_sec` | cost |
| `exit_status` | how the agent loop ended (`completed` / `max_turns` / …) |

## Reproducing

Generated with [Quant-Tuner](https://github.com/pearsonkyle/Quant-Tuner); see
`docs/ternary_qat.md` for the end-to-end pipeline and
`src/quant_tuner/datasets/` for the exact builder used to publish this.
