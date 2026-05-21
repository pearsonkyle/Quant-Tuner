#!/usr/bin/env python3
"""Tool-call accuracy benchmark for a single GGUF quant.

Runs the model against a holdout JSONL (one session per line, see
``scripts/build_toolcall_holdout.py``) in two passes:

  1. Per-turn pass — for each ground-truth assistant tool-call turn, replay
     the prior context and score: tool selection, parameter accuracy, schema
     validity.
  2. Rollout pass — once per session, run the full agentic loop feeding in
     recorded tool results, and record whether the model sustains the
     conversation.

Either point at a running llama-server via ``--base-url`` or let the script
spawn one for you (default).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.eval.toolcall import (
    DEFAULT_SYSTEM_PROMPT,
    Sampling,
    run_toolcall_eval,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path,
                   help="Path to GGUF (required unless --base-url given without spawning)")
    p.add_argument("--base-url",
                   help="OpenAI-compatible base URL; skips server spawn if set")
    p.add_argument("--holdout", type=Path, required=True,
                   help="JSONL holdout corpus, one session per line")
    p.add_argument("--out", type=Path, required=True,
                   help="CSV to append the summary row to")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Directory for per-turn jsonl + server log (default: alongside --out)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None,
                   help="Nucleus sampling threshold (OpenAI-standard)")
    p.add_argument("--top-k", type=int, default=None,
                   help="Top-k filter (llama.cpp extension)")
    p.add_argument("--min-p", type=float, default=None,
                   help="Min-probability threshold (llama.cpp extension)")
    p.add_argument("--presence-penalty", type=float, default=None,
                   help="OpenAI-standard presence penalty")
    p.add_argument("--repetition-penalty", type=float, default=None,
                   help="llama.cpp `repeat_penalty` (1.0 = disabled)")
    p.add_argument("--seed", type=int, default=None,
                   help="Per-server sampling seed")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--max-turns-per-session", type=int, default=8,
                   help="Cap on per-turn evaluations per session")
    p.add_argument("--rollout-max-turns", type=int, default=12)
    p.add_argument("--skip-rollout", action="store_true")
    p.add_argument("--server-startup-timeout", type=float, default=120.0)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                   help="System prompt injected when a session lacks one. "
                        "Pass empty string to disable injection.")
    p.add_argument("--no-stop-on-fail", action="store_true",
                   help="Continue scoring further turns even after the model picks "
                        "the wrong tool (or no tool).")
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
    stem = (args.model.stem if args.model else "remote")
    per_turn_log = log_dir / f"toolcall_{stem}_{ts}.jsonl"
    server_log = log_dir / f"server_{stem}_{ts}.log" if args.model and not base_url else None

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
    if any(v is not None for v in (args.top_p, args.top_k, args.min_p,
                                   args.presence_penalty, args.repetition_penalty,
                                   args.seed)):
        print(f"sampling: temperature={args.temperature} + "
              f"{ {k: v for k, v in asdict(sampling).items() if v is not None and k != 'temperature'} }",
              flush=True)

    summary = run_toolcall_eval(
        holdout=args.holdout,
        model_path=args.model if not base_url else None,
        base_url=base_url,
        sampling=sampling,
        model_label=args.model.name if args.model else "remote",
        max_turns_per_session=args.max_turns_per_session,
        rollout_max_turns=args.rollout_max_turns,
        skip_rollout=args.skip_rollout,
        stop_on_fail=not args.no_stop_on_fail,
        system_prompt=args.system_prompt or None,
        ctx=args.ctx,
        ngl=args.ngl,
        server_log_path=server_log,
        server_startup_timeout=args.server_startup_timeout,
        chat_template_kwargs=(args.chat_template_kwargs or None),
        per_turn_log=per_turn_log,
        progress=True,
    )

    # Print summary.
    n = summary.n_turns
    n_sel = round(summary.tool_selection_acc * n) if n else 0
    n_schema = round(summary.schema_valid_rate * n) if n else 0
    n_roll = len(summary.rollouts)
    print("\n" + "=" * 60)
    print(f"Model: {summary.model}")
    print(f"  Tool selection accuracy:   {n_sel}/{n}  ({summary.tool_selection_acc:.3f})")
    print(f"  Param accuracy (mean):     {summary.param_acc_mean:.3f}")
    print(f"  Schema-valid rate:         {n_schema}/{n}  ({summary.schema_valid_rate:.3f})")
    if n_roll:
        n_done = round(summary.rollout_complete_rate * n_roll)
        n_match = round(summary.rollout_tool_set_match_rate * n_roll)
        print(f"  Rollouts completed:        {n_done}/{n_roll}")
        print(f"  Tool-set match (rollout):  {n_match}/{n_roll}")
    if len(summary.by_source) > 1:
        print("\n  Per-source breakdown:")
        for src in sorted(summary.by_source):
            s = summary.by_source[src]
            print(f"    {src:10s} n={s['n']:3d}  "
                  f"sel={s['tool_selection_acc']:.2f}  "
                  f"param={s['param_acc_mean']:.2f}  "
                  f"schema={s['schema_valid_rate']:.2f}")

    print(f"\n  Per-turn log:              {per_turn_log}")

    # Append CSV row.
    row = {
        "model": os.path.basename(summary.model),
        "n_sessions": summary.n_sessions,
        "n_turns": summary.n_turns,
        "tool_selection_acc": summary.tool_selection_acc,
        "param_acc_mean": summary.param_acc_mean,
        "schema_valid_rate": summary.schema_valid_rate,
        "rollout_complete_rate": summary.rollout_complete_rate,
        "rollout_tool_set_match_rate": summary.rollout_tool_set_match_rate,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p if sampling.top_p is not None else "",
        "top_k": sampling.top_k if sampling.top_k is not None else "",
        "min_p": sampling.min_p if sampling.min_p is not None else "",
        "presence_penalty": sampling.presence_penalty if sampling.presence_penalty is not None else "",
        "repetition_penalty": sampling.repetition_penalty if sampling.repetition_penalty is not None else "",
        "seed": sampling.seed if sampling.seed is not None else "",
        "timestamp": ts,
    }
    new_file = not args.out.exists()
    with args.out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f"  Appended to:               {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
