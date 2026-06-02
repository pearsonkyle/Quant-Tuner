#!/usr/bin/env python3
"""Run the MMMU multimodal eval on one vision GGUF via llama-server.

Either pass ``--model`` + (optionally) ``--mmproj`` to spawn a server here, or
pass ``--base-url`` to reuse an already-running llama-server.

The script loads the holdout JSON produced by ``scripts/build_mmmu_holdout.py``
(images live in a sibling ``images/`` directory), runs every sample through the
model, and appends one summary row to ``--out`` (CSV).

The llama-server must be started with multimodal support.  For models that
require a separate projector GGUF use ``--mmproj``.  For models loaded with
``-hf`` (which auto-download the projector) the flag can be omitted and the
server is started without an explicit ``--mmproj`` argument.

Usage examples
--------------
# Spawn server automatically
uv run python scripts/eval_mmmu.py \\
    --model  out/mmmu/gguf/Qwen2-VL-7B-Q4_K_M.gguf \\
    --mmproj out/mmmu/gguf/mmproj-Qwen2-VL-7B-Q4_K_M.gguf \\
    --holdout out/mmmu/holdout.json \\
    --out out/mmmu/results.csv

# Reuse an already-running server (e.g. started with llama-server -hf …)
uv run python scripts/eval_mmmu.py \\
    --base-url http://127.0.0.1:8080 \\
    --holdout out/mmmu/holdout.json \\
    --out out/mmmu/results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.mmmu import render_summary, run_mmmu_eval
from quant_tuner.eval.toolcall import Sampling


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- model / server ---
    server_grp = p.add_argument_group("server")
    server_grp.add_argument(
        "--model", type=Path,
        help="Path to the text-model GGUF.  Required unless --base-url is given.",
    )
    server_grp.add_argument(
        "--mmproj", type=Path, dest="mmproj",
        help=(
            "Path to the multimodal projector GGUF (--mmproj flag forwarded to "
            "llama-server).  Required for models that ship the projector as a "
            "separate file.  Omit for -hf models that bundle the projector."
        ),
    )
    server_grp.add_argument(
        "--base-url",
        help=(
            "OpenAI-compatible base URL of an already-running llama-server.  "
            "Mutually exclusive with --model/--mmproj."
        ),
    )
    server_grp.add_argument("--ctx", type=int, default=8192,
                             help="Context window size (default: 8192).")
    server_grp.add_argument("--ngl", type=int, default=99,
                             help="GPU layers (default: 99 = all).")
    server_grp.add_argument("--server-startup-timeout", type=float, default=180.0,
                             help="Seconds to wait for llama-server health check.")

    # --- data ---
    data_grp = p.add_argument_group("data")
    data_grp.add_argument(
        "--holdout", type=Path, required=True,
        help="MMMU holdout JSON produced by build_mmmu_holdout.py.",
    )
    data_grp.add_argument(
        "--out", type=Path, required=True,
        help="CSV file to append the summary row to.",
    )
    data_grp.add_argument(
        "--log-dir", type=Path, default=None,
        help="Directory for per-sample JSONL log and server log (default: alongside --out).",
    )

    # --- sampling ---
    smp_grp = p.add_argument_group("sampling")
    smp_grp.add_argument("--temperature", type=float, default=0.0)
    smp_grp.add_argument("--top-p", type=float, default=None)
    smp_grp.add_argument("--top-k", type=int, default=None)
    smp_grp.add_argument("--min-p", type=float, default=None)
    smp_grp.add_argument("--presence-penalty", type=float, default=None)
    smp_grp.add_argument("--repetition-penalty", type=float, default=None)
    smp_grp.add_argument("--seed", type=int, default=None)
    smp_grp.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max completion tokens per sample (default: 512).",
    )
    smp_grp.add_argument(
        "--chat-template-kwargs", type=str, default=None,
        help=(
            "JSON string forwarded to llama-server's --chat-template-kwargs "
            "(e.g. '{\"enable_thinking\":false}' to disable Qwen3 reasoning).  "
            "Omit to use the server default."
        ),
    )

    return p


def main() -> int:
    args = _build_arg_parser().parse_args()

    # --- validate args ---
    if not args.holdout.exists():
        print(f"ERROR: holdout not found: {args.holdout}", file=sys.stderr)
        return 1
    if args.base_url and args.model:
        print("ERROR: --base-url and --model are mutually exclusive.", file=sys.stderr)
        return 1
    if not args.base_url and not args.model:
        print("ERROR: one of --model or --base-url is required.", file=sys.stderr)
        return 1
    if args.model and not args.model.exists():
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        return 1
    if args.mmproj and not args.mmproj.exists():
        print(f"ERROR: mmproj file not found: {args.mmproj}", file=sys.stderr)
        return 1

    base_url = args.base_url
    if base_url:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

    log_dir = args.log_dir or args.out.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.model.stem if args.model else "remote"
    per_sample_log = log_dir / f"mmmu_{stem}_{ts}.jsonl"
    server_log = (
        log_dir / f"server_mmmu_{stem}_{ts}.log"
        if args.model and not base_url else None
    )

    sampling = Sampling(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    print("=== MMMU evaluation ===", flush=True)
    if args.model:
        print(f"  model:    {args.model}", flush=True)
        if args.mmproj:
            print(f"  mmproj:   {args.mmproj}", flush=True)
    else:
        print(f"  base-url: {base_url}", flush=True)
    print(f"  holdout:  {args.holdout}", flush=True)
    print(f"  sampling: T={sampling.temperature}  top_p={sampling.top_p}"
          f"  top_k={sampling.top_k}", flush=True)

    summary = run_mmmu_eval(
        holdout_path=args.holdout,
        model_path=args.model if not base_url else None,
        mmproj_path=args.mmproj if not base_url else None,
        base_url=base_url,
        sampling=sampling,
        model_label=args.model.name if args.model else "remote",
        ctx=args.ctx,
        ngl=args.ngl,
        server_log_path=server_log,
        server_startup_timeout=args.server_startup_timeout,
        chat_template_kwargs=(args.chat_template_kwargs or None),
        per_sample_log=per_sample_log,
        progress=True,
    )

    print()
    print(render_summary(summary))
    print(f"\n  Per-sample log: {per_sample_log}")

    # --- append CSV row ---
    row: dict = {
        "model": summary.model,
        "n_total": summary.n_total,
        "n_correct": summary.n_correct,
        "accuracy": summary.accuracy,
        "n_mc": summary.n_mc,
        "n_mc_correct": summary.n_mc_correct,
        "mc_accuracy": (summary.n_mc_correct / summary.n_mc) if summary.n_mc else "",
        "n_open": summary.n_open,
        "n_open_correct": summary.n_open_correct,
        "open_accuracy": (
            (summary.n_open_correct / summary.n_open) if summary.n_open else ""
        ),
        "n_unparseable": summary.n_unparseable,
        "temperature": sampling.temperature,
        "seed": sampling.seed if sampling.seed is not None else "",
        "timestamp": ts,
    }
    for subj in sorted(summary.by_subject):
        safe = subj.replace(" ", "_")
        row[f"{safe}_acc"] = summary.by_subject[subj]["accuracy"]
        row[f"{safe}_n"] = summary.by_subject[subj]["n"]

    new_file = not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f"  Appended to:    {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
