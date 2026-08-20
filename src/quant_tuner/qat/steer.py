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

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
              cap_p: float = 0.5, k: int | Sequence[int] = 1,
              bank: dict | None = None) -> RepBatch:
        """``k`` = how many identical (call -> identical result) rounds precede the
        decision point (a sequence assigns ks round-robin across the n contexts).
        k=1 is the original first-repeat context; measured on anchor7
        (scripts/measure_repeat_prob.py, 2026-08-20): P(repeat) at k=1 sits at ~0.33
        — under the 0.5 cap, so v1's hinge never fired — and ESCALATES 0.33->0.52
        over k=1..5 on trained latents while vanilla stays flat at ~0.35. Train at
        k=2-5 with the cap at vanilla's own level: penalize the escalation, not
        repetition per se."""
        rng = random.Random(seed)
        ks = [k] * n if isinstance(k, int) else [ks_ for ks_ in k]
        ctxs, reps = [], []
        for i in range(n):
            sysm = rng.choice([x for x in SYSTEMS if x != PROBE_SYSTEM])
            if bank is not None:
                # REAL material (build_rep_bank.py): anchor8 achieved the synthetic
                # objective (escalation inverted, below vanilla at every k) while the
                # real episode looped 56x — the synthetic states don't transfer.
                task = rng.choice(bank["tasks"])
                call, result = rng.choice(bank["pairs"])
            else:
                task = rng.choice(TASKS).format(mod=rng.choice(MODS))
                cmd = rng.choice(COMMANDS).format(mod=rng.choice(MODS))
                call = ('<tool_call>\n{"name": "bash", "arguments": '
                        f'{{"command": "{cmd}"}}\n</tool_call>')
                result = rng.choice(REP_RESULTS)
            lead = rng.choice(LEADS)
            prefix = tok.apply_chat_template(
                [{"role": "system", "content": sysm},
                 {"role": "user", "content": task}],
                tools=TOOLS, tokenize=False, add_generation_prompt=True)
            # ONE result, identical every round — the loop trap
            round_ = (call + "<|im_end|>\n"
                      + "<|im_start|>user\n<tool_response>\n" + result
                      + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n")
            ctx = prefix + lead + "\n" + round_
            for _r in range(ks[i % len(ks)] - 1):  # later rounds: bare re-issue, same result
                ctx += round_
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

    @classmethod
    def from_harvest(cls, tok, path, *, n: int = 4, seed: int = 31,
                     cap_p: float = 0.5) -> RepBatch:
        """Contexts harvested from REAL agent episodes (build_rep_traj_contexts.py).

        These are the states where the loop actually lives: measured on anchor9, the
        constructed contexts read 0.08-0.15 at any depth while the full-prefix real
        state reads 0.96+, and truncating the prefix collapses it — so rows keep the
        whole history and the caller amortizes cost by applying every Nth step."""
        import json as _json
        rows = [_json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        if not rows:
            raise ValueError(f"empty harvest file {path}")
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:n]
        for r in rows:
            assert PROBE_SENTENCE not in r["ctx"] and PROBE_USER not in r["ctx"]
        enc_c = [tok(r["ctx"], add_special_tokens=False).input_ids for r in rows]
        enc_r = [tok(r["rep"], add_special_tokens=False).input_ids for r in rows]
        pad = tok.pad_token_id or 0
        ids, attn, span = _pad_rows(enc_c, enc_r, pad)
        return cls(ids, attn, span, math.log(cap_p))


def _pad_rows(enc_c, enc_r, pad):
    import torch as _t
    n = len(enc_c)
    L = max(len(c) + len(r) for c, r in zip(enc_c, enc_r, strict=True))
    ids = _t.full((n, L), pad, dtype=_t.long)
    attn = _t.zeros((n, L), dtype=_t.long)
    span = _t.zeros((n, 2), dtype=_t.long)
    for i, (c, r) in enumerate(zip(enc_c, enc_r, strict=True)):
        row = c + r
        off = L - len(row)
        ids[i, off:] = _t.tensor(row)
        attn[i, off:] = 1
        span[i, 0] = off + len(c)
        span[i, 1] = off + len(row)
    return ids, attn, span


def rep_fingerprint(batch: RepBatch) -> str:
    """Content hash of the contexts a RepKD table was captured on. The trainer refuses a
    table whose fingerprint differs from its freshly built batch — a KL against logits
    captured on OTHER contexts would train silently against the wrong states (same
    failure class as the KD table's corpus-fingerprint guard)."""
    h = hashlib.sha256()
    h.update(batch.ids.cpu().numpy().tobytes())
    h.update(batch.span.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


@dataclass
class RepKD:
    """Teacher top-K logprobs at every span position of a RepBatch (captured offline by
    scripts/capture_rep_teacher.py). The KL turns the repetition hinge from "don't be
    certain about the verbatim repeat" into "match what the teacher does in this state"
    — the hinge suppresses one action but supplies no alternative; the teacher's
    distribution is the principled alternative."""

    idx: torch.Tensor        # [P, K] support token ids (concatenated over rows)
    logp: torch.Tensor       # [P, K] teacher logprobs at those ids
    tail: torch.Tensor       # [P] teacher log-mass outside the support
    row_off: torch.Tensor    # [m+1] row i owns positions row_off[i]:row_off[i+1]
    fingerprint: str
    teacher: str

    def to(self, device) -> RepKD:
        return RepKD(self.idx.to(device), self.logp.to(device), self.tail.to(device),
                     self.row_off, self.fingerprint, self.teacher)

    @classmethod
    def load(cls, path: str | Path, batch: RepBatch) -> RepKD:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        fp = rep_fingerprint(batch)
        if blob["fingerprint"] != fp:
            raise ValueError(
                f"rep-KD table {path} was captured on different contexts "
                f"(table {blob['fingerprint']}, batch {fp}) — regenerate with "
                f"scripts/capture_rep_teacher.py using the run's exact "
                f"--steer-rep-k/--steer-rep-seed/--steer-rep-cap settings")
        want = int((batch.span[:, 1] - batch.span[:, 0]).sum())
        if int(blob["idx"].shape[0]) != want:
            raise ValueError(f"rep-KD table has {blob['idx'].shape[0]} positions, "
                             f"batch spans need {want}")
        return cls(blob["idx"].long(), blob["logp"].float(), blob["tail"].float(),
                   blob["row_off"].long(), blob["fingerprint"], blob.get("teacher", "?"))


def repetition_losses(
    model, batch: RepBatch, kd: RepKD | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
    """One forward over the rep contexts -> (hinge, optional teacher-KL, stats).

    Hinge: one-sided on the mean per-token log-prob of re-issuing the previous command.
    Teacher-forced: position t's logits score token t+1, so the span's tokens are
    predicted by positions span-1 .. span_end-2. Only rows whose mean exceeds the cap
    contribute — the model may still re-run commands, it just cannot be near-certain
    about doing so verbatim right after seeing the result.

    KL (when ``kd`` given): tail-bucket KL(teacher || student) at the same span
    positions, via :func:`quant_tuner.qat.kd_precompute.kd_loss_from_topk`.
    """
    # Trunk first, lm_head ONLY at span positions: a full model() call materializes
    # [n, L, vocab] logits — 9.2 GiB fp32 at n=10 real-material contexts (L~1500,
    # 151k vocab), which OOM'd anchor9 beside the training window. Span positions
    # are ~50/row, so the vocab dim only ever exists on [~550, vocab].
    hidden = model.model(input_ids=batch.ids,
                         attention_mask=batch.attn).last_hidden_state
    means, pos_rows = [], []
    for i in range(batch.ids.shape[0]):
        lo, hi = int(batch.span[i, 0]), int(batch.span[i, 1])
        row_logits = model.lm_head(hidden[i, lo - 1:hi - 1]).float()
        tgt = batch.ids[i, lo:hi]
        lp = torch.log_softmax(row_logits, dim=-1).gather(
            -1, tgt.unsqueeze(-1)).squeeze(-1)
        means.append(lp.mean())
        pos_rows.append(row_logits)
    m = torch.stack(means)
    pen = (m - batch.cap_logp).clamp_min(0.0).mean()
    stats = {"rep_pen": float(pen.detach()),
             "rep_p_mean": float(m.detach().exp().mean())}
    kl = None
    if kd is not None:
        from quant_tuner.qat.kd_precompute import kd_loss_from_topk
        student = torch.cat(pos_rows, dim=0)          # [P, V], row order = batch order
        kl = kd_loss_from_topk(student, kd.idx, kd.logp, tail=kd.tail, temp=1.0)
        stats["rep_kl"] = float(kl.detach())
    return pen, kl, stats


def repetition_loss(model, batch: RepBatch) -> tuple[torch.Tensor, dict[str, float]]:
    """Hinge-only wrapper (the original interface)."""
    pen, _, stats = repetition_losses(model, batch)
    return pen, stats
