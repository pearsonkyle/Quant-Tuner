"""Record builder for the SWE agentic-trajectory dataset.

Turns the raw eval artifacts written by ``eval/swebench.py``

    <workspace>/trajectories/<model>/<instance_id>.traj.json    # OpenAI-Agents item list
    <workspace>/trajectories/<model>/<instance_id>.result.json  # patch + grade + metrics

into flat, chat-template-ready records. The message flattening is shared with the QAT corpus
builder (:func:`quant_tuner.qat.corpus.trajectory_to_messages`) so the published dataset and
what we actually trained on cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Default source: the Ornith distillation-generation run.
DEFAULT_TRAJ_DIRS = [
    REPO / "out" / "swe-rebench" / "ornith-distill-gen"
    / "trajectories" / "Ornith-1.0-9B-Q5_K_M",
]

# metrics copied verbatim from <instance>.result.json onto each record
METRIC_FIELDS = [
    "repo", "resolved", "patch_produced", "patch_chars", "exit_status",
    "n_model_calls", "tools_used", "tool_errors",
    "prompt_tokens", "completion_tokens", "total_tokens", "wall_sec",
    "n_fail_to_pass", "n_fail_to_pass_passed", "n_pass_to_pass", "n_pass_to_pass_passed",
]


def iter_trajectory_records(
    *,
    resolved_only: bool = True,
    traj_dirs: list[Path] | None = None,
) -> Iterator[dict]:
    """Yield one record per graded trajectory, newest-stable ordering (sorted by instance)."""
    from quant_tuner.qat.corpus import BASH_TOOL, SWE_SYSTEM_PROMPT, trajectory_to_messages

    for traj_dir in (traj_dirs or DEFAULT_TRAJ_DIRS):
        traj_dir = Path(traj_dir)
        if not traj_dir.exists():
            continue
        for tf in sorted(traj_dir.glob("*.traj.json")):
            inst = tf.name[: -len(".traj.json")]
            rf = traj_dir / f"{inst}.result.json"
            if not rf.exists():          # never graded (errored instance)
                continue
            res = json.loads(rf.read_text())
            if resolved_only and not res.get("resolved"):
                continue
            blob = json.loads(tf.read_text())
            messages = trajectory_to_messages(blob["messages"])
            if not any(m["role"] == "assistant" for m in messages):
                continue
            yield {
                "instance_id": inst,
                "source_model": traj_dir.name,
                "messages": messages,
                "tools": [BASH_TOOL],
                "system_prompt": SWE_SYSTEM_PROMPT,
                "submission": res.get("submission", ""),
                "n_messages": len(messages),
                "n_tool_calls": sum(len(m.get("tool_calls") or []) for m in messages),
                **{k: res.get(k) for k in METRIC_FIELDS},
            }
