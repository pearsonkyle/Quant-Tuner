#!/usr/bin/env python
"""Anchor prompts: put the FAILING CONTEXT CLASS in distribution.

Every anchor run holds P(stop) at corpus positions (the anchor penalty stays ~0.01)
while the probe's context class — short agentic exchanges ending right after a tool
call — degrades in deepening waves. The corpus can't defend a context class it barely
contains: its after-tool-call stops sit deep inside 32k windows of long sessions. This
builds a few hundred SHORT synthetic conversations of exactly that class (varied tasks,
lead sentences, tools, commands; single- and two-call shapes), masks them with the
production labeler (stop token supervised), packs them at the training window, and
merges them into an existing corpus blob under a new fingerprint.

THE PROBE STAYS HELD OUT: generation asserts the probe's exact USER task and SENTENCE
never appear — training on the measurement would turn the probe into a memorized
answer instead of a canary.

    python scripts/build_anchor_prompts.py \
        --base out/exp-058/fixed/corpus_ourssft_32768.pt \
        --out  out/exp-058/fixed/corpus_ourssft_ap_32768.pt [--n 480]

The merged blob needs a fresh KD table (fingerprint-keyed):
    python scripts/kd_precompute.py --corpus <out> --include-ids 151645 ...
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from quant_tuner.qat.corpus import (  # noqa: E402
    corpus_fingerprint,
    load_tokenizer,
    masked_ids_for_session,
    pack,
)
from quant_tuner.qat.stop_probe import SENTENCE as PROBE_SENTENCE  # noqa: E402
from quant_tuner.qat.stop_probe import USER as PROBE_USER  # noqa: E402

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
    "Profile {mod} and remove the quadratic loop the issue mentions.",
    "A recent commit broke serialization in {mod}. Bisect and repair it.",
    "Document the public functions of {mod} and fix any signature drift.",
    "The linter flags unused imports across {mod}. Clean them up.",
    "Exceptions from {mod} lose their tracebacks. Preserve them.",
    "Make {mod} handle unicode paths on Windows.",
]
MODS = ["utils", "core/config", "io/readers", "api/session", "parsers", "db/models",
        "cli", "scheduler", "auth/tokens", "metrics"]
LEADS = [
    "I'll start by looking at the repository layout.",
    "First, let me find the relevant module.",
    "Let me inspect the failing code path.",
    "I need to see the current implementation before changing anything.",
    "Let me check the test suite structure first.",
    "I'll reproduce the problem before touching the code.",
    "Let me search for where this is defined.",
    "First step: locate the module and its tests.",
]
SECOND_LEADS = [
    "The listing shows where the module lives. Let me open it.",
    "Now let me look at the implementation itself.",
    "That confirms the layout. Next, the failing function.",
    "Good, the tests are under tests/. Let me run the relevant ones.",
]
COMMANDS = [
    "ls -la", "git status", "grep -rn '{mod}' src/ | head -20",
    "find . -name '*.py' -path '*{mod}*'", "cat src/{mod}.py | head -50",
    "python -m pytest tests/ -k '{mod}' -x -q", "git log --oneline -5",
    "sed -n '1,40p' src/{mod}.py", "rg 'def ' src/{mod}.py | head",
    "python -c 'import {mod}'",
]
RESULTS = [
    "src/\ntests/\nsetup.py\nREADME.md",
    "total 24\ndrwxr-xr-x src\ndrwxr-xr-x tests\n-rw-r--r-- setup.py",
    "src/{mod}.py:12: def load(path):\nsrc/{mod}.py:40: def dump(obj, path):",
    "============ 1 failed, 12 passed in 3.41s ============",
    "abc1234 fix parser edge case\ndef5678 add cache layer",
]
BASH_TOOL = {"type": "function", "function": {
    "name": "bash",
    "description": "Run a shell command in the repository.",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string",
                                              "description": "the command"}},
                   "required": ["command"]}}}


def gen_conversations(n: int, seed: int = 7) -> list[list[dict]]:
    rng = random.Random(seed)
    convs = []
    for _ in range(n):
        mod = rng.choice(MODS)
        task = rng.choice(TASKS).format(mod=mod)
        lead = rng.choice(LEADS)
        cmd = rng.choice(COMMANDS).format(mod=mod)
        call = {"id": f"call_{rng.randrange(16**8):08x}", "type": "function",
                "function": {"name": "bash", "arguments": {"command": cmd}}}
        msgs = [
            {"role": "system", "content": rng.choice(SYSTEMS)},
            {"role": "user", "content": task},
            {"role": "assistant", "content": lead, "tool_calls": [call]},
        ]
        if rng.random() < 0.4:                    # two-call shape: result -> second stop
            res = rng.choice(RESULTS).format(mod=mod)
            cmd2 = rng.choice(COMMANDS).format(mod=mod)
            call2 = {"id": f"call_{rng.randrange(16**8):08x}", "type": "function",
                     "function": {"name": "bash", "arguments": {"command": cmd2}}}
            msgs += [
                {"role": "tool", "content": res, "tool_call_id": call["id"]},
                {"role": "assistant", "content": rng.choice(SECOND_LEADS),
                 "tool_calls": [call2]},
            ]
        convs.append(msgs)
    return convs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="corpus blob to extend")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tok = load_tokenizer()
    convs = gen_conversations(args.n, args.seed)

    stream_ids: list[int] = []
    stream_lbl: list[int] = []
    n_stop = 0
    for msgs in convs:
        text = tok.apply_chat_template(msgs, tools=[BASH_TOOL], tokenize=False,
                                       add_generation_prompt=False)
        # the probe is the measurement — it must never be a training sample
        assert PROBE_SENTENCE not in text and PROBE_USER not in text, \
            "generated text collides with the held-out probe"
        ids, lbl = masked_ids_for_session(msgs, tok, tools=[BASH_TOOL], text=text)
        stream_ids += ids
        stream_lbl += lbl
        n_stop += sum(1 for t in lbl if t == 151645)

    base = torch.load(args.base, weights_only=False)
    window = base["ids"].shape[1]
    wins = pack(stream_ids, stream_lbl, window, min_trainable=8)
    if not wins:  # fewer stream tokens than one window: pad the tail with -100 labels
        pad = window - len(stream_ids)
        wins = [{"ids": stream_ids + [tok.pad_token_id or 0] * pad,
                 "labels": stream_lbl + [-100] * pad}]
    ap_ids = torch.tensor([w["ids"] for w in wins], dtype=base["ids"].dtype)
    ap_lbl = torch.tensor([w["labels"] for w in wins], dtype=base["labels"].dtype)

    ids_t = torch.cat([base["ids"], ap_ids])
    lbl_t = torch.cat([base["labels"], ap_lbl])
    blob = dict(base)
    blob["ids"], blob["labels"] = ids_t, lbl_t
    blob["fingerprint"] = corpus_fingerprint(ids_t, lbl_t)
    blob["anchor_prompts"] = {"n_convs": args.n, "seed": args.seed,
                              "n_windows": len(wins), "n_stop_targets": n_stop,
                              "base": str(args.base),
                              "base_fingerprint": base.get("fingerprint")}
    src = blob.get("sources")
    if isinstance(src, list) and len(src) == base["ids"].shape[0]:
        blob["sources"] = src + ["anchor-prompts"] * len(wins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"[ap] {args.n} conversations -> {len(wins)} windows "
          f"({n_stop} supervised stop targets, "
          f"{sum(1 for t in stream_lbl if t != -100)} trainable tokens)")
    print(f"[ap] merged {base['ids'].shape[0]} + {len(wins)} windows -> {args.out} "
          f"(fingerprint {blob['fingerprint']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
