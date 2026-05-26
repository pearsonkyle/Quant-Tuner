"""Build a tool-call holdout for exp-002 from the `test` slice of logtrain.jsonl.

Exp-001 used the `train` slice for calibration and the `holdout` slice for
KLD eval. This script uses the unused `test` slice for tool-call rollout
eval — disjoint by construction (same seed=42, 80/10/10 split as exp-001).

Output format matches what eval/toolcall.py:_load_sessions expects: one JSON
record per line with `session_id`, `tools`, `messages`, plus a few extras.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data import ingest, split

LOGTRAIN = REPO / "logtrain.jsonl"
OUT = REPO / "out" / "exp-002" / "toolcall_holdout.jsonl"


def _extract_tools(msgs: list[dict]) -> list[dict] | None:
    """Tools may live on the first system or first user message (per source)."""
    for m in msgs:
        if isinstance(m.get("tools"), list):
            return m["tools"]
    return None


def main() -> int:
    sessions = ingest.load_sessions(LOGTRAIN)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    print(f"  {len(sessions)} sessions after exp-001 filter")

    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
    test = splits["test"]
    print(f"  split sizes: train={len(splits['train'])} "
          f"test={len(test)} holdout={len(splits['holdout'])}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped_no_tools = 0
    skipped_no_call = 0
    with OUT.open("w") as f:
        for s in test:
            msgs = ingest.normalize_messages(s.get("messages", []))
            tools = _extract_tools(msgs)
            if not tools:
                skipped_no_tools += 1
                continue
            if not any(m.get("role") == "assistant" and m.get("tool_calls") for m in msgs):
                skipped_no_call += 1
                continue
            rec = {
                "session_id": s.get("session_id") or s.get("id"),
                "source": s.get("source"),
                "score": s.get("score"),
                "tools_used": s.get("tools_used", []),
                "tool_call_count": s.get("metrics", {}).get("tool_calls", 0),
                "tools": tools,
                "messages": msgs,
            }
            f.write(json.dumps(rec) + "\n")
            kept += 1

    print(f"  wrote {kept} sessions to {OUT.relative_to(REPO)}")
    if skipped_no_tools or skipped_no_call:
        print(f"  skipped: no_tools={skipped_no_tools} no_assistant_call={skipped_no_call}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
