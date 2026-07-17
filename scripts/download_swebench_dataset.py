#!/usr/bin/env python3
"""Download the FULL nebius/SWE-rebench test split to disk once, as a local jsonl.

Why: the datasets-server ``/rows`` preview API (used by build_swebench_holdout by default)
is rate-limited (HTTP 429) and only serves 100 rows per call, so repeatedly sampling fresh
instances across auto-loop rounds gets throttled. Downloading the parquet shards once
(non-streaming, cached by ``datasets``) makes every subsequent sample a local, offline read.

Writes ``out/external/swe-rebench/all_test.jsonl`` — one slimmed, gradeable instance per line
(same _KEEP_FIELDS as the holdout builder). Idempotent: skips if the file already exists and
is non-empty unless ``--force``. After this, use ``build_swebench_holdout.py --from-local``.

    .venv/bin/python scripts/download_swebench_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_KEEP_FIELDS = [
    "instance_id", "repo", "base_commit", "environment_setup_commit",
    "problem_statement", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS",
    "image_name", "docker_image", "version", "install_config", "meta",
]


def _slim(row: dict) -> dict:
    return {k: row.get(k) for k in _KEEP_FIELDS if k in row}


def _gradeable(row: dict) -> bool:
    return bool(row.get("FAIL_TO_PASS") and (row.get("image_name") or row.get("docker_image")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nebius/SWE-rebench")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path,
                    default=_REPO / "out" / "external" / "swe-rebench" / "all_test.jsonl")
    ap.add_argument("--force", action="store_true", help="re-download even if --out exists")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.stat().st_size > 0 and not args.force:
        n = sum(1 for _ in args.out.open())
        print(f"[download] {args.out} already has {n} instances (use --force to refresh)")
        return 0

    from datasets import load_dataset  # heavy import deferred
    print(f"[download] loading {args.dataset} [{args.split}] (non-streaming; caches all "
          f"parquet shards to disk once)...", flush=True)
    ds = load_dataset(args.dataset, split=args.split)  # downloads + memory-maps locally
    print(f"[download] {len(ds)} rows in the split; slimming + filtering to gradeable...", flush=True)

    n_written = 0
    n_lite = 0
    with args.out.open("w") as f:
        for row in ds:
            row = dict(row)
            if not _gradeable(row):
                continue
            f.write(json.dumps(_slim(row)) + "\n")
            n_written += 1
            if (row.get("meta") or {}).get("is_lite"):
                n_lite += 1
    print(f"[download] wrote {n_written} gradeable instances ({n_lite} lite) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
