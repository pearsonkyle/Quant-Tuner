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

from quant_tuner.qat.dialect import ChatDialect
from quant_tuner.qat.dialect import detect as detect_dialect

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


#: gemma-4 markers. Its tool RESULT is rendered inside the same model turn, so a probe
#: point can sit after one — which Qwen's format has no equivalent of.
_G_TOOL_CALL = '<|tool_call>call:bash{command:<|"|>ls -la<|"|>}<tool_call|>'
_G_TOOL_RESP = ('<|tool_response>response:bash{value:<|"|>src/table.py  tests/<|"|>}'
                '<tool_response|>')
_G_ANSWER = "I fixed it: the error message now names the expected column order."


@dataclass(frozen=True)
class ProbeSpec:
    """Per-family probe text, which points mean what, and the shipped model's readings.

    The probe is only interpretable against a baseline: "sentence_period = 0.95" means
    nothing until you know the shipped model reads 0.009 there. Those references are
    per-model measurements, not constants — `scripts/measure_stop_baseline.py` produces
    them, and a family with none yet prints the raw numbers rather than a false comparison.

    ``diagnostic`` is a position where stopping is WRONG (it should stay low; it rising is
    the early-termination collapse). ``control`` is a position where stopping is RIGHT (it
    should stay high-ish; it falling is the opposite failure — a model that has lost the
    ability to end a turn, i.e. the looping mode). Both are needed: a run can hold one and
    break the other, and only reading them together tells the two apart.
    """

    tool_call: str
    points: list[tuple[str, str]]
    diagnostic: str
    control: str
    #: (diagnostic, control) readings for the untrained model, or None if not yet measured.
    vanilla: tuple[float, float] | None = None


#: Qwen/Bonsai values are the ones the curriculum doc is written against — do not edit
#: them; a different number here silently rewrites every past run's interpretation.
PROBE_SPECS: dict[str, ProbeSpec] = {
    "qwen": ProbeSpec(tool_call=TOOL_CALL, points=PROBE_POINTS,
                      diagnostic=DIAGNOSTIC, control=CONTROL,
                      vanilla=(0.0092, 0.99995)),
    # gemma-4 needs DIFFERENT POINTS, not just different markers, and the reason is
    # structural. Qwen's assistant turn ends at its tool call, so `after_tool_call` reads
    # 0.99995 and is a clean "stopping is right" control. gemma's template instead hands
    # over to the harness there (it emits an opening <|tool_response> as the generation
    # prompt), and the shipped model duly reads **0.00004** — using it as the control would
    # invert the test.
    #
    # Measured on the shipped E4B, the honest position is that gemma-4 has NO sharp stop
    # point: after a complete answer it prefers "\n\n" (0.275) over <turn|>, and
    # `answer_after_tool` at **0.0703** is still the highest of every candidate tried
    # (after_tool_call 0.00004, after_tool_response 0.00021, an answer with no tool use
    # 0.026, mid-answer 0.000). So the control has ~25x of headroom above the diagnostic
    # where Qwen's had ~10^4. It still detects the looping direction — 0.07 -> ~0 is a real
    # signal — but it is a weaker instrument than Qwen's, and a run that moves it should be
    # checked against a trajectory rather than trusted alone.
    "gemma4": ProbeSpec(
        tool_call=_G_TOOL_CALL,
        points=[
            ("start", ""),
            ("mid_sentence", "Let me explore the repository"),
            ("sentence_period", SENTENCE),
            ("sentence_newline", SENTENCE + "\n"),
            ("after_tool_call", SENTENCE + "\n" + _G_TOOL_CALL),
            # stopping is WRONG here: the model should issue the next call (top-1 is
            # <|tool_call> at 0.392 on the shipped model)
            ("after_tool_response", SENTENCE + "\n" + _G_TOOL_CALL + _G_TOOL_RESP),
            # stopping is RIGHT here: the task is done and the turn should end
            ("answer_after_tool",
             SENTENCE + "\n" + _G_TOOL_CALL + _G_TOOL_RESP + _G_ANSWER),
        ],
        diagnostic="sentence_period",
        control="answer_after_tool",
        vanilla=(0.002744, 0.070316),
    ),
}


@dataclass
class StopProbe:
    """Prebuilt token ids for each probe point — built once, reused every call."""

    stop_id: int
    prompts: list[tuple[str, torch.Tensor]]
    dialect: str = "qwen"

    @property
    def spec(self) -> ProbeSpec:
        return PROBE_SPECS[self.dialect]

    @property
    def diagnostic(self) -> str:
        """Name of the point where stopping is WRONG — what --probe-abort watches."""
        return self.spec.diagnostic

    @property
    def control(self) -> str:
        """Name of the point where stopping is RIGHT."""
        return self.spec.control

    @classmethod
    def build(cls, tok, dialect: ChatDialect | None = None) -> StopProbe:
        if dialect is None:
            dialect = detect_dialect(tok)
        spec = PROBE_SPECS[dialect.name]
        stop_piece = dialect.stop_piece
        stop_id = tok.convert_tokens_to_ids(stop_piece)
        if stop_id is None or stop_id < 0:
            raise ValueError(f"{stop_piece} is not in this tokenizer's vocabulary")
        # add_generation_prompt=True gives the assistant/model turn header, which is what
        # makes `start` mean "the model has produced nothing yet".
        prefix = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER}],
            tools=TOOLS, tokenize=False, add_generation_prompt=True,
        )
        prompts = []
        for name, suffix in spec.points:
            # Special tokens in `suffix` (the tool-call markers) must encode to their real
            # single ids, which is transformers' default for in-text special tokens.
            ids = tok(prefix + suffix, add_special_tokens=False,
                      return_tensors="pt").input_ids
            prompts.append((name, ids))
        return cls(stop_id=stop_id, prompts=prompts, dialect=dialect.name)

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


def format_line(probs: dict[str, float], dialect: str = "qwen") -> str:
    """One compact log line; the diagnostic and control are named so a reader of the raw
    log does not need to remember which of the points matters.

    Which points those ARE is per family (see :class:`ProbeSpec`), and the "vs vanilla"
    comparison is printed only for a family whose baseline has been measured — quoting
    Bonsai's 0.0092 next to a gemma reading would be worse than printing nothing."""
    spec = PROBE_SPECS.get(dialect, PROBE_SPECS["qwen"])
    parts = " ".join(f"{k}={v:.4f}" for k, v in probs.items())
    d = probs.get(spec.diagnostic, float("nan"))
    c = probs.get(spec.control, float("nan"))
    if spec.vanilla is None:
        return (f"{parts}  [diagnostic {spec.diagnostic}={d:.4f}; "
                f"control {spec.control}={c:.4f}; no measured {dialect} baseline — "
                f"run scripts/measure_stop_baseline.py]")
    return (f"{parts}  [diagnostic {spec.diagnostic}={d:.4f} vs vanilla {spec.vanilla[0]}; "
            f"control {spec.control}={c:.4f} vs vanilla {spec.vanilla[1]}]")
