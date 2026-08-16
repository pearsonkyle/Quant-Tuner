#!/usr/bin/env python3
"""Convert a packed QAT corpus (``.pt``) to ``jsonl.gz`` — same windows, no torch needed.

``build_sft_qat_corpus.py`` saves a torch blob because that is what ``train_qat`` loads.
This writes the identical windows as line-delimited JSON so another framework (or another
machine without this repo checked out) can train on them.

    python scripts/export_qat_corpus_jsonl.py \\
        --pt out/exp-058/sft_corpus_universal_32768.pt \\
        --out out/exp-058/sft_corpus_universal_32768.jsonl.gz

One line per window::

    {"i": 0, "source": "logs-agents", "n_trainable": 8123, "density": 0.248,
     "ids": [...], "labels": [...]}          # labels: -100 where masked, else == ids

``labels`` is written in full rather than as a mask. It is redundant — a label is either
-100 or the id at that position — but it is what a trainer consumes directly, and the
repetition costs almost nothing after gzip.

The blob's scalar metadata (window, fingerprint, per-source counts, the settings it was
built with) goes to a sidecar ``.meta.json`` so every line of the jsonl stays uniform.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=Path, required=True, help="packed corpus from build_sft_qat_corpus.py")
    ap.add_argument("--out", type=Path, required=True, help="destination .jsonl.gz")
    ap.add_argument("--compresslevel", type=int, default=6)
    args = ap.parse_args()

    import torch

    blob = torch.load(args.pt, map_location="cpu", weights_only=False)
    ids, labels = blob["ids"], blob["labels"]
    names = list(blob.get("source_names") or [])
    win_src = blob.get("window_source")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = ids.shape[0]
    with gzip.open(args.out, "wt", compresslevel=args.compresslevel) as fh:
        for i in range(n_rows):
            row_ids = ids[i].tolist()
            row_lbl = labels[i].tolist()
            n_train = sum(1 for x in row_lbl if x != -100)
            src = names[int(win_src[i])] if win_src is not None and names else None
            fh.write(json.dumps({
                "i": i,
                "source": src,
                "n_trainable": n_train,
                "density": round(n_train / len(row_lbl), 4),
                "ids": row_ids,
                "labels": row_lbl,
            }) + "\n")

    meta = {k: v for k, v in blob.items() if k not in ("ids", "labels", "window_source")}
    meta["windows"] = n_rows
    meta["source_names"] = names
    meta["jsonl"] = str(args.out)
    meta_path = args.out.with_suffix("").with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    mb = args.out.stat().st_size / 1024**2
    print(f"[export] {n_rows} windows of {blob['window']} -> {args.out} ({mb:.1f} MiB gz)")
    print(f"[export] metadata -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
