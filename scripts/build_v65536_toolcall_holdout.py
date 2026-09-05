#!/usr/bin/env python3
"""Build a tool-call eval holdout from the v65536 pack's doubly-held-out rows.

``exp060_build_toolcall_holdout_from_sft.py`` keys on a ``split`` field that the
v2 pack does not carry: the trainer computes its split at load time, and stage 0
and stage 1 were handed different packs, so neither stage's own split is a
holdout for the other. The rows that ARE clean for both were identified
separately and stored as row indices into ``sft-131072.jsonl``:

    /workspace/logs/clean_holdout_idx.json   269 rows over 34 sources

That file is the only thing standing between this eval and the same
contamination that flattened stage 1's ``eval_loss``: 293 of the 300 rows the
trainer's own eval scores are in stage 0's train split. Do not substitute a
fresh random sample for it.

Of the 269, 112 carry a ``tools`` schema; the ones that additionally contain an
assistant turn with ``tool_calls`` are the scorable sessions -- a session with
no call contributes no scorable turn and only dilutes the session count.

Emits the schema ``quant_tuner.eval.toolcall`` expects, one session per line:

    {"session_id", "source", "stratum", "tools", "messages"}

``source`` is the corpus source, so the aggregator can report tool-call accuracy
per tool family rather than as one blended number. ``stratum`` is the 32K-token
length band -- the same boundary the bpb work split on (167 short / 102 long
over the full 269), so the two measurements can be read side by side. Keep the
bands in separate runs: a long-stratum prefix needs ``--ctx 131072`` to be
scored at the length stage 1 was trained for, and that is far slower per turn
than the short band needs.

    python scripts/build_v65536_toolcall_holdout.py \
        --out out/e4b-v65536/eval/toolcall_holdout.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 3.2 chars/token on this corpus; 32,768 tokens is the band boundary the clean
# holdout was split on, and this reproduces its 167/102 exactly.
CHARS_PER_TOKEN = 3.2
BAND_TOKENS = 32_768


def _j(x):
    """Rows store `messages` entries and `tools` as JSON strings, not objects."""
    return json.loads(x) if isinstance(x, str) else x


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", type=Path,
                   default=Path("/workspace/sft-corpus-v2-rw-v65536/sft-131072.jsonl"))
    p.add_argument("--clean-idx", type=Path,
                   default=Path("/workspace/logs/clean_holdout_idx.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stratum", choices=["short", "long", "both"], default="both",
                   help="Emit only one length band (default: both, tagged)")
    p.add_argument("--min-tool-turns", type=int, default=1,
                   help="Drop sessions with fewer assistant tool_calls turns")
    p.add_argument("--max-sessions", type=int, default=0,
                   help="Cap the session count, chosen round-robin across "
                        "sources (0 = keep all). A blended tool-call number is "
                        "meaningless if one source dominates -- nemotron-swe "
                        "alone is 69 of the 110 -- so trimming for speed has to "
                        "trim the big sources first, not sample uniformly.")
    a = p.parse_args()

    idx = set(json.load(open(a.clean_idx)))
    band_chars = BAND_TOKENS * CHARS_PER_TOKEN

    sessions, skipped_no_tools, skipped_no_calls = [], 0, 0
    with a.pack.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i not in idx:
                continue
            row = json.loads(line)
            if not row.get("tools"):
                skipped_no_tools += 1
                continue
            msgs = [_j(m) for m in row["messages"]]
            n_calls = sum(1 for m in msgs
                          if m.get("role") == "assistant" and m.get("tool_calls"))
            if n_calls < a.min_tool_turns:
                skipped_no_calls += 1
                continue
            stratum = "long" if len(json.dumps(msgs)) >= band_chars else "short"
            if a.stratum != "both" and stratum != a.stratum:
                continue
            sessions.append({
                "session_id": f"clean-{i}",
                "source": row.get("source", "?"),
                "stratum": stratum,
                "tools": _j(row["tools"]),
                "messages": msgs,
                "n_truth_tool_turns": n_calls,
            })

    if a.max_sessions and len(sessions) > a.max_sessions:
        # Within a source, take the sessions with the most ground-truth calls
        # first: each scored turn costs a prefill, and a session carrying six
        # calls yields more scored turns per prefill than one carrying a single
        # call. Then round-robin across sources until the cap is reached.
        by_source: dict[str, list] = {}
        for sess in sessions:
            by_source.setdefault(sess["source"], []).append(sess)
        for v in by_source.values():
            v.sort(key=lambda x: -x["n_truth_tool_turns"])
        picked, queues = [], list(by_source.values())
        while len(picked) < a.max_sessions and any(queues):
            for q in queues:
                if q and len(picked) < a.max_sessions:
                    picked.append(q.pop(0))
            queues = [q for q in queues if q]
        sessions = picked

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as fh:
        for s in sessions:
            fh.write(json.dumps(s) + "\n")

    by_src: dict[str, int] = {}
    by_band: dict[str, int] = {}
    turns = 0
    for s in sessions:
        by_src[s["source"]] = by_src.get(s["source"], 0) + 1
        by_band[s["stratum"]] = by_band.get(s["stratum"], 0) + 1
        turns += s["n_truth_tool_turns"]
    print(f"wrote {len(sessions)} sessions -> {a.out}")
    print(f"  {turns} ground-truth tool-call turns")
    print(f"  skipped: {skipped_no_tools} without a tools schema, "
          f"{skipped_no_calls} without a tool call")
    print(f"  bands: " + ", ".join(f"{k} {v}" for k, v in sorted(by_band.items())))
    print("  sources:")
    for k, v in sorted(by_src.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<28} {v:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
