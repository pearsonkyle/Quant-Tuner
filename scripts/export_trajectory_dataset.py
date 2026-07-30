#!/usr/bin/env python3
"""Export the harvested agentic SWE trajectories as a portable, chat-template-ready JSONL.

The raw trajectories live under a SWE-rebench eval workspace as one pair of files per issue:

    out/swe-rebench/<run>/trajectories/<model>/<instance_id>.traj.json    # the conversation
    out/swe-rebench/<run>/trajectories/<model>/<instance_id>.result.json  # patch + grade + metrics

``.traj.json`` is the raw OpenAI-Agents item list (function_call / function_call_output /
message items), which is awkward to consume elsewhere. This flattens each one into standard
chat ``messages`` (system / user / assistant-with-tool_calls / tool) using the SAME converter
the QAT corpus uses, and attaches the tool schema + grading metadata, so a downstream project
can do:

    rec = json.loads(line)
    text = tokenizer.apply_chat_template(rec["messages"], tools=rec["tools"], tokenize=False)

Each line is one complete agentic trajectory: an issue, then many steps of
(assistant tool_call -> tool output), ending in a submitted patch. These are multi-step
problem-solving sessions, not single-turn QA.

    .venv/bin/python scripts/export_trajectory_dataset.py                       # all graded
    .venv/bin/python scripts/export_trajectory_dataset.py --resolved-only       # verified wins
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.qat.corpus import (  # noqa: E402
    BASH_TOOL,
    SWE_SYSTEM_PROMPT,
    trajectory_to_messages,
)

# metrics copied verbatim from <instance>.result.json onto each record
_METRIC_FIELDS = [
    "repo", "resolved", "patch_produced", "patch_chars", "exit_status",
    "n_model_calls", "tools_used", "tool_errors",
    "prompt_tokens", "completion_tokens", "total_tokens", "wall_sec",
    "n_fail_to_pass", "n_fail_to_pass_passed", "n_pass_to_pass", "n_pass_to_pass_passed",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path, action="append", default=None,
                    help="trajectory dir(s); default = the Ornith distill-gen run")
    ap.add_argument("--resolved-only", action="store_true",
                    help="keep only trajectories whose hidden tests PASSED (verified wins)")
    ap.add_argument("--patched-only", action="store_true",
                    help="keep only trajectories that produced a non-empty patch")
    ap.add_argument("--out", type=Path,
                    default=REPO / "out" / "datasets" / "swe_agentic_trajectories.jsonl")
    args = ap.parse_args()

    traj_dirs = args.traj_dir or [
        REPO / "out" / "swe-rebench" / "ornith-distill-gen"
        / "trajectories" / "Ornith-1.0-9B-Q5_K_M"
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_skipped = 0
    n_resolved = 0
    total_steps = 0
    with args.out.open("w") as fh:
        for traj_dir in traj_dirs:
            traj_dir = Path(traj_dir)
            for tf in sorted(traj_dir.glob("*.traj.json")):
                inst = tf.name[: -len(".traj.json")]
                rf = traj_dir / f"{inst}.result.json"
                if not rf.exists():
                    n_skipped += 1          # never graded (crashed/errored instance)
                    continue
                res = json.loads(rf.read_text())
                if args.resolved_only and not res.get("resolved"):
                    n_skipped += 1
                    continue
                if args.patched_only and not res.get("patch_produced"):
                    n_skipped += 1
                    continue

                blob = json.loads(tf.read_text())
                messages = trajectory_to_messages(blob["messages"])
                if not any(m["role"] == "assistant" for m in messages):
                    n_skipped += 1
                    continue

                rec = {
                    "instance_id": inst,
                    "source_model": traj_dir.name,
                    "messages": messages,
                    "tools": [BASH_TOOL],
                    "system_prompt": SWE_SYSTEM_PROMPT,
                    # the final git diff the agent submitted, and how it was graded
                    "submission": res.get("submission", ""),
                    "n_messages": len(messages),
                    "n_tool_calls": sum(len(m.get("tool_calls") or []) for m in messages),
                    **{k: res.get(k) for k in _METRIC_FIELDS},
                }
                fh.write(json.dumps(rec) + "\n")
                n_written += 1
                n_resolved += bool(res.get("resolved"))
                total_steps += rec["n_tool_calls"]

    size_mb = args.out.stat().st_size / 1024 **2 if args.out.exists() else 0
    print(f"[dataset] wrote {n_written} trajectories ({n_resolved} resolved/verified), "
          f"{n_skipped} skipped -> {args.out} ({size_mb:.1f} MB)")
    if n_written:
        print(f"[dataset] mean {total_steps / n_written:.1f} tool calls per trajectory")
    print("[dataset] use: tokenizer.apply_chat_template(rec['messages'], tools=rec['tools'])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
