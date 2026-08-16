# Chat templates

Candidate replacements for a model's shipped `chat_template.jinja`, kept here so a
template swap is reviewable and A/B-testable rather than an in-place edit of a
checkpoint.

## qwen3_8_safe_v2.jinja

From the r/LocalLLaMA "Fixed/improved Jinja chat template for Qwen 3.8" thread
(template_version `qwen3.8-safe-v2`). Its stated design goal is to stay
**byte-identical to the trained original** for well-formed inputs — the author's
argument being that Qwen models underperform subtly when the rendered prompt
deviates from what they were trained on — while fixing crashes and silent
corruption on malformed or non-native inputs.

Fixes that we independently reproduced against the stock Qwen3.8 template:

- **JSON-string tool arguments crash the stock template** with
  `TypeError: Can only get item pairs from a mapping`. That is the OpenAI wire
  shape (`function.arguments` as a JSON string). Note vLLM pre-parses those to
  dicts (`vllm/entrypoints/chat_utils.py`), so this does **not** affect a vLLM
  deployment — it bites raw-transformers and other harnesses.

Claimed fixes we have not needed yet: malformed multimodal list items silently
becoming vision tokens, a leading tool message emitting a tool turn with no
`<|im_start|>user`, and validation of unknown roles / later system messages.

**Before adopting, verify byte-identity on real traffic** with
`scripts/ab_chat_template.py` — if the render is byte-identical on our corpus,
the swap cannot change quality and is a pure robustness win, and no behavioural
A/B is required.
