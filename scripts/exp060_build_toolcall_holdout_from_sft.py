"""Build a tool-call eval holdout from ``sft.jsonl.gz``'s holdout split.

``scripts/build_toolcall_holdout.py`` samples from the on-disk CLI logs, which are
local-only and absent on this box. But ``sft.jsonl.gz`` carries the same conversations in
full, with a ``split`` field, so the holdout slice (disjoint from the ``train`` slice the
imatrix was calibrated on) gives a genuine tool-call holdout.

Emits the schema ``eval/toolcall.py`` expects — one session per line:

    {"session_id": ..., "tools": [...], "messages": [...]}

Only sessions that actually contain assistant tool calls are kept: a session with no call
contributes no scorable turn and would just dilute the session count.

    PYTHONPATH=src .venv/bin/python scripts/exp060_build_toolcall_holdout_from_sft.py \\
        --out out/exp-060-32k/eval/toolcall_holdout.jsonl -n 25
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data import ingest

SEED = 42


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sft", type=Path, default=Path("/workspace/sft.jsonl.gz"))
    p.add_argument("--split", default="holdout")
    p.add_argument("--sources", nargs="+", default=["logs", "logs-agents"])
    p.add_argument("-n", "--n-sessions", type=int, default=25)
    p.add_argument("--out", type=Path,
                   default=REPO / "out/exp-060-32k/eval/toolcall_holdout.jsonl")
    a = p.parse_args()

    rows = [json.loads(ln) for ln in gzip.open(a.sft, "rt")]
    pool = [r for r in rows
            if r.get("split") == a.split and r.get("source") in a.sources]

    kept = []
    for r in pool:
        msgs = ingest.normalize_messages(r.get("messages") or [])
        ingest.coerce_tool_call_arguments(msgs)
        n_calls = sum(1 for m in msgs
                      if m.get("role") == "assistant" and m.get("tool_calls"))
        if not n_calls or not r.get("tools"):
            continue
        kept.append({
            "session_id": r.get("id") or f"sft-{len(kept)}",
            "tools": r["tools"],
            "messages": msgs,
            "_n_calls": n_calls,
        })

    rng = random.Random(SEED)
    rng.shuffle(kept)
    picked = kept[: a.n_sessions]
    total_calls = sum(s.pop("_n_calls") for s in picked)
    for s in kept[a.n_sessions:]:
        s.pop("_n_calls", None)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for s in picked:
            f.write(json.dumps(s) + "\n")

    print(f"pool ({a.split}, {a.sources}): {len(pool)} sessions")
    print(f"  with tool calls + schemas : {len(kept)}")
    print(f"  sampled                   : {len(picked)}")
    print(f"  assistant tool calls      : {total_calls}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
