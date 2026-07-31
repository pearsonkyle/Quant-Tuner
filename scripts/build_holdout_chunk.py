"""Build a held-out text chunk for AWQ α cross-validation (exp-017/018).

Source: logtrain test-slice sessions 5-24 (raw user+assistant text).
Disjoint by construction from both:
  - calibration corpus (mixed8k.txt = train slice, same seed=42 split), AND
  - smoke eval (out/smoke/holdout.jsonl, test-slice sessions 1-3).

Note: an earlier draft of this builder also pulled from project source files
(CLAUDE.md, README.md) under the "different distribution" rationale. That
turned out to be contaminated — some train-slice sessions in logtrain quote
or discuss the project documentation verbatim, so those source files appear
inside mixed8k. Using them as held-out would silently weaken the signal.
Pure logtrain-test text is the only clean source we have here.

Output: out/holdout_chunks/cv_1k.txt (~6k tokens of disjoint conversation
text, large enough to use the first 256 tokens as the AWQ proxy cache).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data import ingest, split

LOGTRAIN = REPO / "logtrain.jsonl"
HOLDOUT_USED_BY_SMOKE = REPO / "out" / "smoke" / "holdout.jsonl"
OUT = REPO / "out" / "holdout_chunks" / "cv_1k.txt"

TARGET_CHARS = 24000  # ~6k tokens; AWQ proxy uses first 256 tokens of activations


def _logtrain_test_text(exclude_session_ids: set[str]) -> str:
    """Concatenate raw user+assistant text from logtrain test slice sessions
    starting at index 5 (extra buffer past the smoke eval set, which used
    indices 1-3)."""
    sessions = ingest.load_sessions(LOGTRAIN)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
    test = splits["test"]
    print(f"  logtrain test slice: {len(test)} sessions", file=sys.stderr)

    out_parts: list[str] = []
    total = 0
    skipped_excluded = 0
    for i, s in enumerate(test):
        sid = s.get("session_id") or s.get("id")
        if sid in exclude_session_ids:
            skipped_excluded += 1
            continue
        if i < 5:  # extra buffer past smoke eval indices
            continue
        msgs = ingest.normalize_messages(s.get("messages", []))
        for m in msgs:
            # Skip system/tool messages — system prompts are shared boilerplate
            # across all train+test+holdout slices, so they appear in mixed8k
            # via OTHER sessions and would silently contaminate the held-out.
            if m.get("role") not in {"user", "assistant"}:
                continue
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if not isinstance(content, str) or not content.strip():
                continue
            out_parts.append(content.strip())
            total += len(content)
            if total >= TARGET_CHARS:
                break
        if total >= TARGET_CHARS:
            break
    print(
        f"  logtrain text: ~{total} chars from {len(out_parts)} segments "
        f"(skipped {skipped_excluded} smoke-overlap sessions)",
        file=sys.stderr,
    )
    return "\n\n".join(out_parts)


def main() -> int:
    smoke_ids: set[str] = set()
    if HOLDOUT_USED_BY_SMOKE.exists():
        for line in HOLDOUT_USED_BY_SMOKE.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec.get("session_id") or rec.get("id")
            if sid:
                smoke_ids.add(sid)
        print(
            f"  excluding {len(smoke_ids)} smoke-eval session ids from holdout chunk",
            file=sys.stderr,
        )

    text = _logtrain_test_text(smoke_ids)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)}  ({len(text)} chars total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
