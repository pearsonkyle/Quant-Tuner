"""Reconstruct exp-060's calibration corpus from the transferred ``corpus.cal.jsonl.gz``.

``universal.build`` cannot run on this box: 42% of the calibration corpus comes from
``datasets/agent-logs/data/*.jsonl.gz``, which is local-only real captured usage and is
not published anywhere. The corpus was therefore transferred prebuilt as a JSONL of the
packed windows, and this script turns it back into the two files
``exp060_quants_qwen38.py`` expects in ``out/<run>/corpora/``:

* ``corpus.cal.txt``      — the windows concatenated exactly as ``split.write_corpus``
  writes them (``text + "\\n\\n"``, verbatim, no rstrip).
* ``corpora_audit.json``  — the ``step("corpora", ...)`` sentinel. ``step`` is
  existence-based, so writing it here makes the quants script SKIP ``universal.build``
  instead of failing on the missing log datasets.

The audit is a stub only in the sense that it carries the three fields the quants script
actually reads (``calibration.total_tokens``, ``.token_share``,
``.tool_calls.tool_call_marker_total``) — but every value in it is MEASURED from the
reconstructed corpus with the repo's own scanners, not copied from a previous run.

    PYTHONPATH=src .venv/bin/python scripts/exp060_inject_cal_corpus.py \\
        --jsonl /workspace/corpus.cal.jsonl.gz
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data.universal import tool_call_marker_counts

# Mirrors UniversalConfig defaults — the corpus was PACKED for these; see the ctx note in
# exp060_quants_qwen38.py. Numbers produced at a different ctx are not comparable.
CTX = 8192
CTX_HEADROOM = 692
SEED = 42


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=Path, default=Path("/workspace/corpus.cal.jsonl.gz"))
    p.add_argument("--run", default="exp-060")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    a = p.parse_args()

    out = REPO / "out" / a.run / "corpora"
    out.mkdir(parents=True, exist_ok=True)

    opener = gzip.open if a.jsonl.suffix == ".gz" else open
    recs = [json.loads(ln) for ln in opener(a.jsonl, "rt")]
    recs.sort(key=lambda r: r["i"])
    assert [r["i"] for r in recs] == list(range(len(recs))), \
        "window indices are not a 0..n-1 permutation — corpus is incomplete"
    for r in recs:
        if r.get("n_chars") is not None:
            assert r["n_chars"] == len(r["text"]), f"window {r['i']} n_chars disagrees"

    # Exactly split.write_corpus: chunk + "\n\n", verbatim, no rstrip, no supplement.
    cal = out / "corpus.cal.txt"
    with cal.open("w") as f:
        for r in recs:
            f.write(r["text"] + "\n\n")
    # newline="" — universal-newline translation would silently rewrite the \r that agent
    # tool output carries, exactly as universal.scan_special_tokens guards against.
    with cal.open(newline="") as f:
        text = f.read()
    n_chars = sum(len(r["text"]) for r in recs)
    assert len(text) == n_chars + 2 * len(recs), "reconstruction length mismatch"

    from transformers import AutoTokenizer

    from quant_tuner.data.template_check import (
        KNOWN_TOOL_CALL_MARKERS,
        KNOWN_TOOL_RESPONSE_MARKERS,
    )

    tok = AutoTokenizer.from_pretrained(a.model)

    # Per-source token counts, so token_share is real rather than a char-share stand-in.
    by_source: dict[str, int] = collections.Counter()
    total_tokens = 0
    for r in recs:
        n = len(tok(r["text"], add_special_tokens=False)["input_ids"])
        by_source[r.get("source", "unknown")] += n
        total_tokens += n

    calls = tool_call_marker_counts(text, KNOWN_TOOL_CALL_MARKERS)
    resps = tool_call_marker_counts(text, KNOWN_TOOL_RESPONSE_MARKERS)

    audit = {
        "builder": "exp060_inject_cal_corpus.py (prebuilt corpus reconstructed; "
                   "universal.build not runnable — agent-logs datasets are local-only)",
        "seed": SEED,
        "ctx": CTX,
        "window_cap": max(512, CTX - CTX_HEADROOM),
        "ctx_note": "corpus was PACKED for ctx=8192; imatrix and eval must use the same "
                    "ctx or the numbers are not comparable",
        "source_jsonl": str(a.jsonl),
        "calibration": {
            "path": str(cal),
            "bytes": cal.stat().st_size,
            "chars": len(text),
            "windows": len(recs),
            "total_tokens": total_tokens,
            "per_source_tokens": dict(by_source),
            "token_share": {s: round(n / total_tokens, 4)
                            for s, n in by_source.most_common()},
            "tool_calls": {
                "tool_call_markers": calls,
                "tool_call_marker_total": sum(calls.values()),
                "tool_response_markers": resps,
                "tool_response_marker_total": sum(resps.values()),
            },
        },
    }
    (out / "corpora_audit.json").write_text(json.dumps(audit, indent=2))

    c = audit["calibration"]
    print(f"wrote {cal}")
    print(f"  windows      : {c['windows']:,}")
    print(f"  chars        : {c['chars']:,}")
    print(f"  bytes        : {c['bytes']:,}")
    print(f"  total_tokens : {c['total_tokens']:,}")
    print(f"  token_share  : {c['token_share']}")
    print(f"  tool calls   : {c['tool_calls']['tool_call_marker_total']:,} "
          f"{c['tool_calls']['tool_call_markers']}")
    print(f"  tool resps   : {c['tool_calls']['tool_response_marker_total']:,}")
    print(f"wrote {out / 'corpora_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
