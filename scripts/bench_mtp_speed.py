"""Compare llama-server decode speed with MTP speculative decoding on vs off.

Spawns ``llama-server`` twice against the same GGUF: first with
``--spec-type draft-mtp --spec-draft-n-max <n>``, then with no spec flags at
all (true baseline). For each configuration we fire a handful of chat
completions and read the ``predicted_per_second`` field from llama-server's
per-response ``timings`` block.

Note: the effective draft depth is capped by the number of nextn layers in the
GGUF (``qwen35.nextn_predict_layers``). For Qwen3.6-27B this is 1, so
``--spec-draft-n-max`` values above 1 give no additional benefit.

Usage:

    uv run python scripts/bench_mtp_speed.py \\
        --model out/q5_k_s_qwen3_6_mtp/gguf/Q5_K_S-imatrix.gguf \\
        --reps 5 --n-max 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.eval.server import running_server  # noqa: E402


PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number using memoization. Include a brief docstring.",
    "Explain what a B-tree is and when you would use one instead of a hash table. Be concrete.",
    "Given a list of integers, write a Rust function that returns the longest strictly-increasing subsequence in O(n log n).",
    "Summarize the difference between TCP and UDP in two short paragraphs, then give one example use case for each.",
    "Write a SQL query that returns, for each user, their three most recent orders along with the order total.",
]


def run_one(base_url: str, prompt: str, max_tokens: int) -> dict:
    """Fire one chat completion and return the response JSON (incl. ``timings``)."""
    r = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "stream": False,
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def bench_config(
    model_path: Path,
    *,
    label: str,
    spec_type: str | None,
    spec_draft_n_max: int | None,
    reps: int,
    max_tokens: int,
    ctx: int,
    chat_template_kwargs: str | None,
    log_dir: Path,
) -> dict:
    """Spawn server with the given spec-decode config, run ``reps`` prompts, summarize."""
    log_path = log_dir / f"llama-server.{label}.log"
    print(f"\n=== {label}  (spec_type={spec_type}, spec_draft_n_max={spec_draft_n_max}) ===")
    print(f"  server log: {log_path}")

    rates: list[float] = []
    n_drafted = 0
    n_accepted = 0

    with running_server(
        model_path,
        ctx=ctx,
        log_path=log_path,
        chat_template_kwargs=chat_template_kwargs,
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max,
        startup_timeout=300.0,
    ) as base_url:
        # Warmup so the first decode doesn't skew the mean.
        print("  warmup...")
        run_one(base_url, "Say hi.", max_tokens=8)

        for i, prompt in enumerate(PROMPTS[:reps]):
            t0 = time.time()
            resp = run_one(base_url, prompt, max_tokens=max_tokens)
            elapsed = time.time() - t0
            timings = resp.get("timings", {}) or {}
            tps = timings.get("predicted_per_second")
            predicted_n = timings.get("predicted_n")
            drafted = timings.get("draft_n", 0) or 0
            accepted = timings.get("draft_n_accepted", 0) or 0
            n_drafted += drafted
            n_accepted += accepted
            print(
                f"  rep {i + 1}/{reps}: "
                f"{tps:.2f} tok/s  ({predicted_n} tokens in {elapsed:.1f}s)  "
                f"draft={accepted}/{drafted}"
                if tps is not None
                else f"  rep {i + 1}/{reps}: timings missing  (elapsed {elapsed:.1f}s)"
            )
            if tps is not None:
                rates.append(tps)

    mean = statistics.mean(rates) if rates else float("nan")
    stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
    accept_rate = (n_accepted / n_drafted) if n_drafted else None
    print(f"  -> mean {mean:.2f} ± {stdev:.2f} tok/s over {len(rates)} reps")
    if accept_rate is not None:
        print(f"  -> draft acceptance: {n_accepted}/{n_drafted} = {accept_rate * 100:.1f}%")
    return {
        "label": label,
        "spec_type": spec_type,
        "spec_draft_n_max": spec_draft_n_max,
        "mean_tps": mean,
        "stdev_tps": stdev,
        "reps": len(rates),
        "draft_accepted": n_accepted,
        "draft_total": n_drafted,
        "accept_rate": accept_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to GGUF with MTP head")
    parser.add_argument("--reps", type=int, default=5, help="Prompts per configuration (max 5)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--n-max", type=int, default=1, help="--spec-draft-n-max for the 'on' run (capped by nextn layers in GGUF; 1 for Qwen3.6-27B)")
    parser.add_argument(
        "--chat-template-kwargs",
        type=str,
        default='{"enable_thinking":false}',
        help='Forwarded to llama-server; pass "" to omit',
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON results path")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2

    log_dir = args.model.parent.parent / "eval" / "mtp_speed"
    log_dir.mkdir(parents=True, exist_ok=True)

    ctk = args.chat_template_kwargs or None

    on = bench_config(
        args.model,
        label=f"spec_n{args.n_max}",
        spec_type="draft-mtp",
        spec_draft_n_max=args.n_max,
        reps=args.reps,
        max_tokens=args.max_tokens,
        ctx=args.ctx,
        chat_template_kwargs=ctk,
        log_dir=log_dir,
    )
    # True baseline: no --spec-type, no --spec-draft-n-max.  Passing
    # --spec-draft-n-max 0 with --spec-type still set is not a valid baseline —
    # llama-server may clamp 0 to its default (3) or keep MTP active.
    off = bench_config(
        args.model,
        label="baseline",
        spec_type=None,
        spec_draft_n_max=None,
        reps=args.reps,
        max_tokens=args.max_tokens,
        ctx=args.ctx,
        chat_template_kwargs=ctk,
        log_dir=log_dir,
    )

    print("\n=== summary ===")
    print(f"  mtp n_max={args.n_max}  {on['mean_tps']:.2f} ± {on['stdev_tps']:.2f} tok/s")
    print(f"  baseline    {off['mean_tps']:.2f} ± {off['stdev_tps']:.2f} tok/s")
    if off["mean_tps"] > 0:
        speedup = on["mean_tps"] / off["mean_tps"]
        print(f"  speedup:    {speedup:.2f}x")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"on": on, "off": off}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
