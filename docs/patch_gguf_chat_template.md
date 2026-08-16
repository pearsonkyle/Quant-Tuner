# Patching the Qwen3.8 chat template in a GGUF release

Runbook for fixing the stock Qwen3.8 chat template in an already-published GGUF repo
(e.g. `pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF`). Written to be pasted into a fresh
agent session; the reasoning behind each step is at the bottom.

The fixed template already exists and is validated:

```
data/chat_templates/qwen3_8_safe_v2.jinja
```

---

## Prompt

```
Patch the Qwen3.8 chat template in my GGUF release repo
(pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF).

The fixed template already exists and is validated:
  /workspace/Quant-Tuner/data/chat_templates/qwen3_8_safe_v2.jinja

It fixes four bugs in Qwen's stock template:
  1. reasoning_effort="high" (the OpenAI-standard value) RAISES -> HTTP 400.
     Ours maps high -> its own instruction (a clause-subset of xhigh).
  2. JSON-string `arguments` (the OpenAI wire shape) crash the render.
  3. A leading `tool` message (no preceding user turn) renders malformed.
  4. A bare string inside a content list silently emits a vision token and
     DISCARDS the text.

Before shipping it, re-verify byte-identity on real traffic — this is the
whole safety argument, since Qwen models are trained on one exact rendered
format:

  PYTHONPATH=src .venv/bin/python scripts/ab_chat_template.py \
      --model out/exp-060/model_extracted \
      --candidate data/chat_templates/qwen3_8_safe_v2.jinja \
      --holdout out/exp-060-32k/eval/toolcall_holdout.jsonl

Expect 382/382 byte-identical. If ANY prefix differs, stop and show me the
diff — a rendering change means a behavioural A/B is required first.

Then patch the GGUFs in place (the template is baked into GGUF metadata at
convert time, so shipping the .jinja alone does nothing for llama.cpp users):

  vendor/llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.py \
      --chat-template-file data/chat_templates/qwen3_8_safe_v2.jinja \
      <in.gguf> <out.gguf>

Do one rung first, verify with gguf_dump.py that tokenizer.chat_template
changed and nothing else did, and confirm llama-server still tool-calls,
before touching the rest. Then re-upload and add a one-line README note.
```

---

## Why each step is there

**Byte-identity is the test that matters, not "is it nicer".** Qwen models are trained on
one exact rendered format, and deviating from it degrades output quality subtly — enough
to move benchmark scores without being visible in manual testing. So the useful question
is *"does the swap change the bytes we actually send?"* Byte-identical on real traffic
⇒ the swap **cannot** change quality, and no behavioural A/B is needed. Any diff ⇒ an A/B
is mandatory before shipping. Measured 2026-08-15: **382/382 identical** on the real
tool-call holdout, while all four edge cases above render differently (i.e. get fixed).

**GGUF differs from the HF path.** In an HF/vLLM repo the template can ship as a loose
`.jinja` file, because vLLM takes `--chat-template`. In a GGUF the template lives *inside*
the file's metadata (`tokenizer.chat_template`), so a loose file changes nothing for
llama.cpp users unless they explicitly pass `--chat-template-file`.
`gguf_new_metadata.py` rewrites that key **without re-quantizing**.

**One rung first.** `gguf_new_metadata.py` rewrites the whole container. Verify with
`gguf_dump.py` that `tokenizer.chat_template` changed and nothing else did, then confirm
`llama-server` still tool-calls, before spending bandwidth on the rest.

## The cheap alternative

Re-uploading four rungs is **~55 GiB**. If that is not worth it: ship
`qwen3_8_safe_v2.jinja` at the repo root and add a README line telling users to pass
`--chat-template-file`. In practice **bug #1 is the only one that bites llama.cpp users**
— #2 is moot on any server that pre-parses tool arguments, and #3/#4 are edge cases — so
this is a fair trade. Both paths are legitimate; pick per how much you care about the
default experience.

## Reference: what the four levels actually do

Verified against both templates by rendering and hashing the full prompt:

| `reasoning_effort` | stock | `safe_v2` |
|:---|:---|:---|
| `xhigh` *(default)* | think carefully + validate assumptions + consider alternatives | same |
| `high` | **raises → HTTP 400** | own instruction (xhigh minus the exploration clause) |
| `medium` | injects **nothing** — native reasoning | same |
| `low` | keep thinking brief and focused | same |

`enable_thinking: false` ships a pre-closed `<think></think>` so generation starts at the
answer. On llama.cpp these ride in `--chat-template-kwargs` or per-request `extra_body`.

> **Do not assume the levels form a ladder.** Measured over 174 tool-call turns,
> tool-selection accuracy ran 0.437 (`xhigh`) → 0.563 (`off`), but replaying the whole
> sweep moved individual levels by up to **6 turns (3.4pp)**, and `high`/`medium`/`low`
> did **not** order consistently between runs. Only the endpoints are reliable.

## Related

- `scripts/ab_chat_template.py` — the byte-diff harness (real holdout + thread edge cases)
- `data/chat_templates/README.md` — provenance of the candidate template
- `out/exp-060-w4a16-32k/release/README.md` — the W4A16 card, which ships the same template
