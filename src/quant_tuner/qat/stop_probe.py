"""Measure P(<|im_end|>) at fixed positions, on a live torch model.

The GGUF probe (`scripts/probe_stop_prob.py`) can only run on an exported model, so the
termination pathology was always discovered hours after the run that caused it — the
sft32k_sw1 ablation spent 11 h to learn that its diagnostic sat at 0.95, a number that
was almost certainly already there by step 50. This runs the same measurement inside the
training loop, so the collapse is visible while it happens and against the code-flip
telemetry that should explain it.

The prompts live HERE and are imported by the GGUF probe, not the other way round: two
copies of the probe text would silently stop being comparable the first time one was
edited, and comparability across the torch and GGUF paths is the entire point.

The two paths are not numerically identical and are not meant to be — this reads the
fp32/bf16 STE-ternarized model on the training device, while the GGUF probe reads a Q2_0
export through llama.cpp with its own kernels. Read the in-training series as a
trajectory, and the GGUF probe as the endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

STOP_PIECE = "<|im_end|>"

SYSTEM = "You are a helpful coding assistant with access to tools."
USER = (
    "There is a bug in the repository: the error message for mismatched columns "
    "does not say what order was expected. Please investigate and fix it."
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "the command"}
                },
                "required": ["command"],
            },
        },
    }
]

# Kept verbatim from the sft32k run's own output, so the probe lands on the exact
# boundary where that run emitted <|im_end|>.
SENTENCE = "Let me explore the repository structure and understand the bug."
TOOL_CALL = (
    '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>'
)

# (name, text appended to the generation prefix). `after_tool_call` is the CONTROL: a
# high value there is correct, and a run that drops it has lost the ability to stop where
# stopping is right, which is a different failure from stopping too early.
PROBE_POINTS: list[tuple[str, str]] = [
    ("start", ""),
    ("mid_sentence", "Let me explore the repository"),
    ("sentence_period", SENTENCE),
    ("sentence_newline", SENTENCE + "\n"),
    ("after_tool_call", SENTENCE + "\n" + TOOL_CALL),
]

DIAGNOSTIC = "sentence_period"
CONTROL = "after_tool_call"


@dataclass
class StopProbe:
    """Prebuilt token ids for each probe point — built once, reused every call."""

    stop_id: int
    prompts: list[tuple[str, torch.Tensor]]

    @classmethod
    def build(cls, tok) -> StopProbe:
        stop_id = tok.convert_tokens_to_ids(STOP_PIECE)
        if stop_id is None or stop_id < 0:
            raise ValueError(f"{STOP_PIECE} is not in this tokenizer's vocabulary")
        # add_generation_prompt=True gives the "<|im_start|>assistant\n" header, which is
        # what makes `start` mean "the model has produced nothing yet".
        prefix = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER}],
            tools=TOOLS, tokenize=False, add_generation_prompt=True,
        )
        prompts = []
        for name, suffix in PROBE_POINTS:
            # Special tokens in `suffix` (the tool-call markers) must encode to their real
            # single ids, which is transformers' default for in-text special tokens.
            ids = tok(prefix + suffix, add_special_tokens=False,
                      return_tensors="pt").input_ids
            prompts.append((name, ids))
        return cls(stop_id=stop_id, prompts=prompts)

    @torch.no_grad()
    def measure(self, model, device) -> dict[str, float]:
        """P(stop token) at the next position, for each probe point.

        Restores the model's previous train/eval mode: called mid-training, leaving the
        model in eval() would silently disable dropout for the rest of the run.
        """
        was_training = model.training
        model.eval()
        out: dict[str, float] = {}
        try:
            for name, ids in self.prompts:
                logits = model(ids.to(device)).logits[0, -1].float()
                out[name] = float(torch.softmax(logits, dim=-1)[self.stop_id])
        finally:
            if was_training:
                model.train()
        return out


def format_line(probs: dict[str, float]) -> str:
    """One compact log line; the diagnostic and control are named so a reader of the raw
    log does not need to remember which of the five points matters."""
    parts = " ".join(f"{k}={v:.4f}" for k, v in probs.items())
    return (f"{parts}  [diagnostic {DIAGNOSTIC}={probs.get(DIAGNOSTIC, float('nan')):.4f}"
            f" vs vanilla 0.0092; control {CONTROL}="
            f"{probs.get(CONTROL, float('nan')):.4f} vs vanilla 0.99995]")
