#!/usr/bin/env python3
"""Sample an agentic SWE-rebench holdout from ``nebius/SWE-rebench``.

Writes ``out/external/swe-rebench/holdout.jsonl`` — one instance dict per line,
keeping only the fields the agentic eval + grader need (problem statement, gold
patch, gold test patch, FAIL_TO_PASS / PASS_TO_PASS node ids, the Docker image
name, base commit, and a little selection metadata).

The full ``test`` split is ~21k rows / multi-GB; downloading it to pick 10
instances is wasteful (and the streaming reader stalls on large parquet shards),
so this pages Hugging Face's lightweight **datasets-server** ``/rows`` API and
selects locally. Pass ``--use-datasets`` to fall back to ``datasets.load_dataset``
instead.

Default: 10 ``is_lite`` instances, seeded shuffle (seed 42) over the first
``--scan-limit`` rows. Lite + a difficulty cap give a weak 2-bit model a fair
shot at producing gradeable patches; widen with ``--no-lite-only`` /
``--max-difficulty``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Fields we persist (keep the holdout small + reviewable).
_KEEP_FIELDS = [
    "instance_id",
    "repo",
    "base_commit",
    "environment_setup_commit",
    "problem_statement",
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "image_name",
    "docker_image",
    "version",
    "install_config",
    "meta",
]

_DS_SERVER = "https://datasets-server.huggingface.co/rows"


def _slim(row: dict) -> dict:
    return {k: row.get(k) for k in _KEEP_FIELDS if k in row}


def _is_lite(row: dict) -> bool:
    meta = row.get("meta") or {}
    return bool(meta.get("is_lite"))


def _difficulty(row: dict) -> int | None:
    meta = row.get("meta") or {}
    score = (meta.get("llm_score") or {}).get("difficulty_score")
    return int(score) if score is not None else None


def _fetch_rows_page(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    params = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    url = f"{_DS_SERVER}?{params}"
    last_err: Exception | None = None
    # HF's datasets-server rate-limits hard (HTTP 429); it needs minutes of backoff, not
    # seconds. Exponential backoff, honoring Retry-After when present, up to ~8 min total.
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                retry_after = e.headers.get("Retry-After")
                wait = int(retry_after) if (retry_after and retry_after.isdigit()) else min(300, 15 * 2 ** attempt)
            else:
                wait = min(60, 3 * 2 ** attempt)
            time.sleep(wait)
        except Exception as e:  # transient network blips
            last_err = e
            time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(f"datasets-server request failed after retries: {last_err}")


def _scan_via_datasets_server(
    dataset: str, config: str, split: str, *, scan_limit: int
) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    page = 100
    while len(rows) < scan_limit:
        data = _fetch_rows_page(dataset, config, split, offset, min(page, scan_limit - len(rows)))
        batch = data.get("rows", [])
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        offset += len(batch)
        total = data.get("num_rows_total")
        if total is not None and offset >= total:
            break
    return rows


def _scan_via_datasets(dataset: str, split: str, *, scan_limit: int) -> list[dict]:
    from datasets import load_dataset  # heavy import deferred

    ds = load_dataset(dataset, split=split, streaming=True)
    rows: list[dict] = []
    for row in ds:
        rows.append(dict(row))
        if len(rows) >= scan_limit:
            break
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10, help="Number of instances to keep (default: 10)")
    p.add_argument("--dataset", default="nebius/SWE-rebench")
    p.add_argument("--config", default="default")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--scan-limit", type=int, default=400,
        help="How many rows to scan before selecting (default: 400)",
    )
    p.add_argument("--lite-only", dest="lite_only", action="store_true", default=True)
    p.add_argument("--no-lite-only", dest="lite_only", action="store_false")
    p.add_argument(
        "--max-difficulty", type=int, default=None,
        help="Keep only instances with meta.llm_score.difficulty_score <= this",
    )
    p.add_argument(
        "--shuffle", dest="shuffle", action="store_true", default=True,
        help="Seeded shuffle of the candidate pool before taking --n (default: on)",
    )
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    p.add_argument(
        "--use-datasets", action="store_true",
        help="Use datasets.load_dataset(streaming) instead of the datasets-server API",
    )
    p.add_argument(
        "--from-local", type=Path, default=None,
        help="Sample from a local jsonl (from download_swebench_dataset.py) instead of the "
             "rate-limited datasets-server — the whole split, offline, no throttling",
    )
    p.add_argument(
        "--exclude", type=Path, default=None,
        help="A jsonl of instances to EXCLUDE by instance_id (e.g. the eval holdout, so "
             "a training/generation pool stays disjoint from what the student is graded on)",
    )
    p.add_argument("--out", type=Path, default=_REPO / "out" / "external" / "swe-rebench" / "holdout.jsonl")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.from_local:
        rows = [json.loads(ln) for ln in args.from_local.read_text().splitlines() if ln.strip()]
        print(f"Read {len(rows)} rows from local {args.from_local} (no datasets-server)", flush=True)
    else:
        print(f"Scanning up to {args.scan_limit} rows of {args.dataset} [{args.split}]…", flush=True)
        if args.use_datasets:
            rows = _scan_via_datasets(args.dataset, args.split, scan_limit=args.scan_limit)
        else:
            rows = _scan_via_datasets_server(
                args.dataset, args.config, args.split, scan_limit=args.scan_limit
            )
        print(f"  fetched {len(rows)} rows", flush=True)

    candidates = rows
    if args.lite_only:
        candidates = [r for r in candidates if _is_lite(r)]
    if args.max_difficulty is not None:
        candidates = [
            r for r in candidates
            if (_difficulty(r) is not None and _difficulty(r) <= args.max_difficulty)
        ]
    # Every gradeable instance needs FAIL_TO_PASS and an image.
    candidates = [
        r for r in candidates
        if r.get("FAIL_TO_PASS") and (r.get("image_name") or r.get("docker_image"))
    ]
    if args.exclude and args.exclude.exists():
        excluded = {
            json.loads(ln)["instance_id"]
            for ln in args.exclude.read_text().splitlines() if ln.strip()
        }
        before = len(candidates)
        candidates = [r for r in candidates if r["instance_id"] not in excluded]
        print(f"  excluded {before - len(candidates)} instances present in {args.exclude.name}",
              flush=True)
    print(
        f"  {len(candidates)} candidates after filters "
        f"(lite_only={args.lite_only}, max_difficulty={args.max_difficulty})",
        flush=True,
    )
    if not candidates:
        print("ERROR: no candidates matched the filters; widen --scan-limit / relax filters",
              file=sys.stderr)
        return 1

    # Deterministic selection: sort by instance_id, optionally seeded-shuffle, take n.
    candidates.sort(key=lambda r: r["instance_id"])
    if args.shuffle:
        random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.n]

    with args.out.open("w") as f:
        for row in selected:
            f.write(json.dumps(_slim(row)) + "\n")

    print(f"\nWrote {args.out}  ({len(selected)} instances)")
    for row in selected:
        diff = _difficulty(row)
        n_f2p = len(row.get("FAIL_TO_PASS") or [])
        print(f"  {row['instance_id']:45s}  lite={_is_lite(row)!s:5s} diff={diff} f2p={n_f2p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
