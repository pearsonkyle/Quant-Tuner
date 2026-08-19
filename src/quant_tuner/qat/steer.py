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


REP_RESULTS = [
    "src/\ntests/\nsetup.py\nREADME.md",
    "(no output)",
    "grep: no matches found",
    "total 8\ndrwxr-xr-x src\ndrwxr-xr-x tests",
    "============ no tests ran in 0.12s ============",
]


@dataclass
class RepBatch:
    """Contexts ending after (command -> result -> assistant header), with the VERBATIM
    previous command appended as a teacher-forced continuation to score."""

    ids: torch.Tensor        # [m, L] left-padded: context + repeated-command tokens
    attn: torch.Tensor       # [m, L]
    span: torch.Tensor       # [m, 2] start/end (in L coords) of the repeated tokens
    cap_logp: float          # per-token cap; only near-verbatim copying penalized

    def to(self, device) -> RepBatch:
        return RepBatch(self.ids.to(device), self.attn.to(device),
                        self.span.to(device), self.cap_logp)

    @classmethod
    def build(cls, tok, *, n: int = 6, seed: int = 23,
              cap_p: float = 0.5) -> RepBatch:
        rng = random.Random(seed)
        ctxs, reps = [], []
        for _ in range(n):
            sysm = rng.choice([x for x in SYSTEMS if x != PROBE_SYSTEM])
            task = rng.choice(TASKS).format(mod=rng.choice(MODS))
            lead = rng.choice(LEADS)
            cmd = rng.choice(COMMANDS).format(mod=rng.choice(MODS))
            call = ('<tool_call>\n{"name": "bash", "arguments": '
                    f'{{"command": "{cmd}"}}\n</tool_call>')
            prefix = tok.apply_chat_template(
                [{"role": "system", "content": sysm},
                 {"role": "user", "content": task}],
                tools=TOOLS, tokenize=False, add_generation_prompt=True)
            result = rng.choice(REP_RESULTS)
            ctx = (prefix + lead + "\n" + call + "<|im_end|>\n"
                   + "<|im_start|>user\n<tool_response>\n" + result
                   + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n")
            ctxs.append(ctx)
            reps.append(call)                    # the VERBATIM previous command
        for t in ctxs:
            assert PROBE_SENTENCE not in t and PROBE_USER not in t
        enc_c = [tok(t, add_special_tokens=False).input_ids for t in ctxs]
        enc_r = [tok(t, add_special_tokens=False).input_ids for t in reps]
        L = max(len(c) + len(r) for c, r in zip(enc_c, enc_r, strict=True))
        pad = tok.pad_token_id or 0
        ids = torch.full((n, L), pad, dtype=torch.long)
        attn = torch.zeros((n, L), dtype=torch.long)
        span = torch.zeros((n, 2), dtype=torch.long)
        for i, (c, r) in enumerate(zip(enc_c, enc_r, strict=True)):
            row = c + r
            off = L - len(row)                   # left-pad
            ids[i, off:] = torch.tensor(row)
            attn[i, off:] = 1
            span[i, 0] = off + len(c)
            span[i, 1] = off + len(row)
        return cls(ids, attn, span, math.log(cap_p))


def repetition_loss(model, batch: RepBatch) -> tuple[torch.Tensor, dict[str, float]]:
    """One-sided hinge on the mean per-token log-prob of re-issuing the previous command.

    Teacher-forced: position t's logits score token t+1, so the span's tokens are
    predicted by positions span-1 .. span_end-2. Only rows whose mean exceeds the cap
    contribute — the model may still re-run commands, it just cannot be near-certain
    about doing so verbatim right after seeing the result.
    """
    out = model(input_ids=batch.ids, attention_mask=batch.attn)
    logp = torch.log_softmax(out.logits.float(), dim=-1)
    means = []
    for i in range(batch.ids.shape[0]):
        lo, hi = int(batch.span[i, 0]), int(batch.span[i, 1])
        tgt = batch.ids[i, lo:hi]
        lp = logp[i, lo - 1:hi - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        means.append(lp.mean())
    m = torch.stack(means)
    pen = (m - batch.cap_logp).clamp_min(0.0).mean()
    return pen, {"rep_pen": float(pen.detach()),
                 "rep_p_mean": float(m.detach().exp().mean())}
