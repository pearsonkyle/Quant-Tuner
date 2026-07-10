#!/usr/bin/env python
"""Audit what slice of a calibration corpus each calibrator actually reads.

Cheap (tokenizer-only, no model load, no GPU). Run this BEFORE and AFTER
rebuilding corpora to see the head-truncation bug and its fix:

  * OLD ingestion (pre-fix): AWQ read the first ``tokens`` (default 4096)
    tokens of the file; GPTQ the first 32768; the AWQ α-search scored on the
    first ``proxy_tokens`` of the FIRST chunk only. On a corpus with a
    250k-token wiki prefix that meant calibrating on pure wiki.
  * NEW ingestion: the same budgets are strided evenly across the whole
    corpus (``quant_tuner.calibrate._ingest.sample_chunks``).

For each calibrator profile the script reports corpus coverage (span + %)
and how many chat/tool markers fall inside the sampled text vs the corpus
overall — a sampled slice with ~0 markers on a marker-rich corpus is
calibrating on the wrong distribution.

Usage:
  uv run python scripts/audit_calibration_coverage.py \
      --corpus out/corpora/corpus.cal.txt \
      --model out/<run>/model_extracted \
      [--markers '<|im_start|>,<tool_call>'] \
      [--audit out/corpora/corpora_audit.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_MARKERS = "<|im_start|>,<tool_call>"


@dataclass
class Profile:
    name: str
    tokens: int
    ctx: int
    strided: bool  # False = pre-fix head truncation


PROFILES = [
    Profile("awq OLD (head)", tokens=4_096, ctx=4_096, strided=False),
    Profile("awq NEW (strided)", tokens=65_536, ctx=4_096, strided=True),
    Profile("gptq OLD (head)", tokens=32_768, ctx=2_048, strided=False),
    Profile("gptq NEW (strided)", tokens=32_768, ctx=2_048, strided=True),
    Profile("outlier-stats OLD (head)", tokens=4_096, ctx=1_024, strided=False),
    Profile("outlier-stats NEW (strided)", tokens=65_536, ctx=4_096, strided=True),
]


def count_markers(text: str, markers: list[str]) -> dict[str, int]:
    # Whitespace-insensitive: some tokenizers' decode re-spaces around special
    # tokens, which would break an exact substring match.
    squashed = "".join(text.split())
    return {m: squashed.count("".join(m.split())) for m in markers}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True,
                   help="HF model/tokenizer dir (e.g. out/<run>/model_extracted)")
    p.add_argument("--markers", type=str, default=DEFAULT_MARKERS,
                   help="comma-separated substrings that indicate chat/tool "
                        f"content (default {DEFAULT_MARKERS!r})")
    p.add_argument("--audit", type=Path, default=None,
                   help="optional corpora_audit.json to summarize boilerplate share")
    a = p.parse_args()

    import torch  # noqa: F401  (transformers backend)
    from transformers import AutoTokenizer

    from quant_tuner.calibrate._ingest import sample_chunks

    markers = [m for m in a.markers.split(",") if m]
    text = a.corpus.read_text()
    tok = AutoTokenizer.from_pretrained(a.model, fix_mistral_regex=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = ids.shape[0]

    corpus_counts = count_markers(text, markers)
    print(f"corpus: {a.corpus}  ({n:,} tokens)")
    print("  markers in full corpus: "
          + "  ".join(f"{m}×{c}" for m, c in corpus_counts.items()))
    print()
    header = f"{'profile':<26} {'tokens':>8} {'cover%':>7} {'span':>21}  markers-in-sample"
    print(header)
    print("-" * len(header))

    for prof in PROFILES:
        if prof.strided:
            chunks = sample_chunks(ids, prof.ctx, prof.tokens)
            # Reconstruct span from chunk identity: strided always includes
            # first + last eligible chunk, so span ~= whole corpus.
            covered = sum(c.numel() for c in chunks)
            span = f"0..{n - 1}" if len(chunks) > 1 else f"0..{covered - 1}"
        else:
            head = ids[: prof.tokens]
            chunks = [c for c in head.split(prof.ctx) if c.numel() >= 2]
            covered = sum(c.numel() for c in chunks)
            span = f"0..{covered - 1}"
        sample_text = "\n".join(
            tok.decode(c, skip_special_tokens=False) for c in chunks)
        counts = count_markers(sample_text, markers)
        cover = 100.0 * covered / max(n, 1)
        marker_str = "  ".join(f"{m}×{c}" for m, c in counts.items())
        print(f"{prof.name:<26} {covered:>8,} {cover:>6.1f}% {span:>21}  {marker_str}")

    # AWQ α-search proxy slice: pre-fix this was the head of chunk 0 only.
    proxy = 512
    head_text = tok.decode(ids[:proxy], skip_special_tokens=False)
    counts = count_markers(head_text, markers)
    print()
    print(f"awq α-search X OLD = first {proxy} tokens of chunk 0: "
          + "  ".join(f"{m}×{c}" for m, c in counts.items()))
    print("awq α-search X NEW = evenly-spaced rows from EVERY sampled chunk "
          "(follows the strided coverage above)")

    if a.audit is not None and a.audit.exists():
        audit = json.loads(a.audit.read_text())
        w = (audit.get("calibration", {}).get("pack_audit", {}) or {}).get("windowing")
        if w:
            print()
            print(f"pack audit ({a.audit}):")
            for k in ("tool_schema_quota", "unique_tool_schemas",
                      "full_schema_sessions", "schema_stub_sessions",
                      "system_prose_tokens", "boilerplate_tokens",
                      "body_tokens", "tool_turn_token_share",
                      "oversize_windows"):
                if k in w:
                    print(f"  {k}: {w[k]}")
            if "boilerplate_tokens" not in w:
                print("  (pre-fix audit: schema tokens were counted as body — "
                      "tool_turn_token_share is overstated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
