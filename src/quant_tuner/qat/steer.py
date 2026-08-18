"""Termination steering: fold the probe's METRIC FAMILY into the loss.

The probe measures P(<|im_end|>) at fixed positions and, until now, could only
*terminate* a run (probe-abort). This module turns the same quantity into gradient:
an auxiliary batch of short synthetic contexts from the probe's two context classes,
forwarded every step —

* **control contexts** (assistant turn ending right after a tool call): CE toward the
  stop token — direct upward pressure on the "must stay HIGH" face;
* **diagnostic contexts** (mid-sentence / completed sentence): a one-sided hinge on
  ``log P(stop)`` above a cap — downward pressure on the "must stay LOW" face, and no
  pressure at all once below it.

Why this exists: every anchor run held P(stop) at CORPUS positions (the anchor penalty
sat at ~0.01 through entire collapses) while exactly this SHORT context class drifted
in waves. As corpus data the class gets gradient ~3 windows per 613 steps; as a
steering batch it gets it every step, which is matched to a drift that moves every step.

THE PROBE STAYS HELD OUT. Contexts are generated from the same varied vocabulary as
:mod:`scripts.build_anchor_prompts` and construction asserts the probe's exact task and
sentence never appear — training on the measurement would blind the canary. Report
steered runs against all three tiers: the steering family (trained), a disjoint-seed
family (held-out generalization), and the untouched probe.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from quant_tuner.qat.stop_probe import SENTENCE as PROBE_SENTENCE
from quant_tuner.qat.stop_probe import SYSTEM as PROBE_SYSTEM
from quant_tuner.qat.stop_probe import USER as PROBE_USER

SYSTEMS = [
    "You are a helpful coding assistant with access to tools.",
    "You are a software engineering agent. Use the available tools to solve the task.",
    "You are an autonomous coding agent working in a git repository.",
]
TASKS = [
    "The test suite fails on {mod} with an import error. Find out why and fix it.",
    "Add a regression test for the off-by-one in {mod}.",
    "The CLI crashes when given an empty config file. Investigate and patch it.",
    "Refactor the duplicated parsing logic in {mod} into one helper.",
    "Users report the cache in {mod} returns stale entries. Track it down.",
    "The {mod} module logs a deprecation warning on import. Silence it properly.",
    "A recent commit broke serialization in {mod}. Bisect and repair it.",
    "Exceptions from {mod} lose their tracebacks. Preserve them.",
]
MODS = ["utils", "core/config", "io/readers", "api/session", "parsers",
        "db/models", "cli", "scheduler"]
LEADS = [
    "I'll start by looking at the repository layout.",
    "First, let me find the relevant module.",
    "Let me inspect the failing code path.",
    "I need to see the current implementation before changing anything.",
    "Let me check the test suite structure first.",
    "I'll reproduce the problem before touching the code.",
]
COMMANDS = [
    "ls -la", "git status", "grep -rn '{mod}' src/ | head -20",
    "find . -name '*.py' -path '*{mod}*'", "python -m pytest tests/ -k '{mod}' -x -q",
    "sed -n '1,40p' src/{mod}.py",
]
TOOLS = [{"type": "function", "function": {
    "name": "bash",
    "description": "Run a shell command in the repository.",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string",
                                              "description": "the command"}},
                   "required": ["command"]}}}]


@dataclass
class SteerBatch:
    """Padded steering contexts, built once and reused every step."""

    ids: torch.Tensor        # [n, L] left-padded
    attn: torch.Tensor       # [n, L]
    want_stop: torch.Tensor  # [n] bool — CE toward stop vs hinge away from it
    cap_logp: float          # hinge cap for continue rows, in log-prob
    stop_id: int

    def to(self, device) -> SteerBatch:
        return SteerBatch(self.ids.to(device), self.attn.to(device),
                          self.want_stop.to(device), self.cap_logp, self.stop_id)

    @classmethod
    def build(cls, tok, *, n: int = 8, seed: int = 11, stop_id: int = 151645,
              cap_p: float = 0.02, stop_frac: float = 0.5) -> SteerBatch:
        rng = random.Random(seed)
        texts: list[str] = []
        wants: list[bool] = []
        n_stop = max(1, round(n * stop_frac))
        for i in range(n):
            sysm = rng.choice([s for s in SYSTEMS if s != PROBE_SYSTEM])
            task = rng.choice(TASKS).format(mod=rng.choice(MODS))
            lead = rng.choice(LEADS)
            prefix = tok.apply_chat_template(
                [{"role": "system", "content": sysm},
                 {"role": "user", "content": task}],
                tools=TOOLS, tokenize=False, add_generation_prompt=True)
            if i < n_stop:                       # control class: next token = stop
                cmd = rng.choice(COMMANDS).format(mod=rng.choice(MODS))
                call = ('<tool_call>\n{"name": "bash", "arguments": '
                        f'{{"command": "{cmd}"}}\n</tool_call>')
                texts.append(prefix + lead + "\n" + call)
                wants.append(True)
            else:                                # diagnostic class: stop must stay LOW
                suffix = lead if rng.random() < 0.5 else lead.rsplit(" ", 2)[0]
                texts.append(prefix + suffix)
                wants.append(False)
        for t in texts:
            assert PROBE_SENTENCE not in t and PROBE_USER not in t, \
                "steering context collides with the held-out probe"
        encs = [tok(t, add_special_tokens=False).input_ids for t in texts]
        L = max(len(e) for e in encs)
        pad = tok.pad_token_id or 0
        ids = torch.full((n, L), pad, dtype=torch.long)
        attn = torch.zeros((n, L), dtype=torch.long)
        for r, e in enumerate(encs):             # left-pad: last position = decision point
            ids[r, L - len(e):] = torch.tensor(e)
            attn[r, L - len(e):] = 1
        return cls(ids, attn, torch.tensor(wants), math.log(cap_p), stop_id)


def steering_loss(model, batch: SteerBatch) -> tuple[torch.Tensor, dict[str, float]]:
    """CE toward stop on control rows + one-sided hinge above the cap on diagnostic rows.

    One forward over the padded batch; the decision logit is the LAST position of each
    row (left-padding puts it at index -1 for every row). Both terms are means over
    their rows, so the balance does not drift with batch composition.
    """
    out = model(input_ids=batch.ids, attention_mask=batch.attn)
    logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    s = logp[:, batch.stop_id]
    stop_rows, cont_rows = batch.want_stop, ~batch.want_stop
    ce = (-s[stop_rows]).mean() if bool(stop_rows.any()) else s.new_zeros(())
    pen = ((s[cont_rows] - batch.cap_logp).clamp_min(0.0).mean()
           if bool(cont_rows.any()) else s.new_zeros(()))
    loss = ce + pen
    return loss, {"steer_stop_ce": float(ce.detach()),
                  "steer_cont_pen": float(pen.detach()),
                  "steer_p_stop_ctrl": float(s[stop_rows].detach().exp().mean())
                  if bool(stop_rows.any()) else float("nan")}
