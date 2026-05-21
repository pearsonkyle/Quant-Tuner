#!/usr/bin/env python3
"""Run the MMLU-Pro few-shot eval on one quantized GGUF.

Either pass ``--model`` (spawns ``llama-server`` for you) or ``--base-url``
(to reuse a server an outer driver already started). The script loads the
holdout JSON produced by ``scripts/build_mmlu_pro_holdout.py``, runs every
sample, and appends a summary row to ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.mmlu_pro import render_summary, run_mmlu_pro_eval
from quant_tuner.eval.toolcall import Sampling


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path,
                   help="Path to GGUF (required unless --base-url given)")
    p.add_argument("--base-url",
                   help="OpenAI-compatible base URL; skips server spawn if set")
    p.add_argument("--holdout", type=Path, required=True,
                   help="MMLU-Pro holdout JSON (see build_mmlu_pro_holdout.py)")
    p.add_argument("--out", type=Path, required=True,
                   help="CSV to append the summary row to")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Per-sample log + server log directory (default: alongside --out)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--presence-penalty", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Max completion tokens. Reasoning models (Qwen3/DeepSeek) "
                        "need headroom for <think>…</think> before the answer.")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--server-startup-timeout", type=float, default=120.0)
    p.add_argument(
        "--chat-template-kwargs",
        type=str,
        default='{"enable_thinking":false}',
        help='Forwarded to llama-server (default disables Qwen3 reasoning). '
             "Pass empty string to omit.",
    )
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()

    if not args.holdout.exists():
        print(f"ERROR: holdout not found at {args.holdout}", file=sys.stderr)
        return 1
    if not args.base_url:
        if not args.model:
            print("ERROR: --model is required when not using --base-url", file=sys.stderr)
            return 1
        if not args.model.exists():
            print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
            return 1

    base_url = args.base_url.rstrip("/") if args.base_url else None
    if base_url and not base_url.endswith("/v1"):
        base_url += "/v1"

    log_dir = args.log_dir or args.out.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.model.stem if args.model else "remote"
    per_sample_log = log_dir / f"mmlu_pro_{stem}_{ts}.jsonl"
    server_log = (log_dir / f"server_mmlu_{stem}_{ts}.log"
                  if args.model and not base_url else None)

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

    summary = run_mmlu_pro_eval(
        holdout_path=args.holdout,
        model_path=args.model if not base_url else None,
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
    print(f"\n  Per-sample log:    {per_sample_log}")

    # CSV row: one per (model, run) — subject accuracies broken out as columns.
    row: dict = {
        "model": summary.model,
        "n_total": summary.n_total,
        "n_correct": summary.n_correct,
        "accuracy": summary.accuracy,
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
    with args.out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f"  Appended to:       {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
