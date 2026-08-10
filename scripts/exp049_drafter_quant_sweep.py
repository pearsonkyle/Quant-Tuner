"""exp-049: how does QUANTIZING THE MTP DRAFTER affect acceptance?

Fixes the trunk at IQ2_M (the package's lowest-bit text trunk) and swaps the
draft model through a ladder of drafter quants, measuring draft acceptance at
n_max = 1..4. Baseline is the shipped Q8_0 drafter; the lower-bit drafters were
requantized from it (`scripts/exp049_*` / llama-quantize --allow-requantize).

Method is identical to exp-048 (5 mixed coding/reasoning prompts × 200 tokens,
temperature=0.3, thinking off, `--reps` reps with distinct seeds); one rep =
the 5 prompts pooled (sum accepted / sum drafted from server `timings`).

NOTE on drafter quants:
- Q6_K / Q4_K_M / IQ3_M requantize fine (IQ3_M needs no imatrix).
- IQ2_M is intentionally absent: it *requires* an imatrix, and the
  `gemma4-assistant` drafter cannot run standalone in llama-imatrix
  ("requires ctx_other to be set"), so no per-drafter imatrix is producible.

Usage:
    .venv/bin/python scripts/exp049_drafter_quant_sweep.py \
        --reps 3 --out out/exp-049/drafter_quant_sweep.json
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
LLAMA_SERVER = REPO / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server"
PKG = REPO / "uploads/pearsonkyle/gemma-4-31b-it-imatrix-GGUF"
EXP = REPO / "out/exp-049"

# Default trunk for this experiment (override with --trunk LABEL).
DEFAULT_TRUNK = "IQ2_M"

# Drafter ladder: label -> gguf path. Q8_0 is the shipped baseline.
# IQ2_M uses a UNIFORM (synthetic, no-activation-weighting) imatrix — see post:
# the gemma4-assistant arch can't run in llama-imatrix to collect a real one.
DRAFTERS = {
    "Q8_0": PKG / "mtp-gemma-4-31B-it.gguf",
    "Q6_K": EXP / "mtp-gemma-4-31B-it-Q6_K.gguf",
    "Q4_K_M": EXP / "mtp-gemma-4-31B-it-Q4_K_M.gguf",
    "IQ3_M": EXP / "mtp-gemma-4-31B-it-IQ3_M.gguf",
    "IQ2_M": EXP / "mtp-gemma-4-31B-it-IQ2_M.gguf",
}
N_VALUES = [1, 2, 3, 4]

# Identical prompt set to exp-046/047/048.
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number using memoization. Include a brief docstring.",
    "Explain what a B-tree is and when you would use one instead of a hash table. Be concrete.",
    "Given a list of integers, write a Rust function that returns the longest strictly-increasing subsequence in O(n log n).",
    "Summarize the difference between TCP and UDP in two short paragraphs, then give one example use case for each.",
    "Write a SQL query that returns, for each user, their three most recent orders along with the order total.",
]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_health(port: int, proc: subprocess.Popen, timeout: float = 300.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (code {proc.returncode})")
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server not healthy in {timeout}s")


def run_config(trunk: Path, drafter: Path, n: int, reps: int, max_tokens: int,
               ctx: int, base_seed: int, log_path: Path) -> dict:
    port = free_port()
    cmd = [
        str(LLAMA_SERVER),
        "-m", str(trunk),
        "--model-draft", str(drafter),
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(n),
        "--jinja", "-c", str(ctx), "-ngl", "999", "-fa", "on",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    per_rep: list[dict] = []
    with log_path.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        try:
            wait_health(port, proc)
            base = f"http://127.0.0.1:{port}/v1/chat/completions"
            requests.post(base, json={"messages": [{"role": "user", "content": "Say hi."}],
                                      "max_tokens": 8}, timeout=120)  # warmup
            for rep in range(reps):
                seed = base_seed + rep
                drafted = accepted = 0
                for prompt in PROMPTS:
                    r = requests.post(base, json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens, "temperature": 0.3, "seed": seed,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }, timeout=600)
                    r.raise_for_status()
                    t = r.json().get("timings", {}) or {}
                    drafted += t.get("draft_n", 0) or 0
                    accepted += t.get("draft_n_accepted", 0) or 0
                rate = (accepted / drafted) if drafted else None
                per_rep.append({"rep": rep, "seed": seed, "draft_total": drafted,
                                "draft_accepted": accepted, "accept_rate": rate})
                rs = f"{rate*100:.1f}%" if rate is not None else "—"
                print(f"      rep {rep} (seed {seed}): {rs}  ({accepted}/{drafted})", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    rates = [r["accept_rate"] for r in per_rep if r["accept_rate"] is not None]
    mean = statistics.fmean(rates) if rates else None
    stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
    sem = (stdev / (len(rates) ** 0.5)) if rates else None
    tot_d = sum(r["draft_total"] for r in per_rep)
    tot_a = sum(r["draft_accepted"] for r in per_rep)
    return {
        "n_max": n,
        "reps": per_rep,
        "mean_accept_rate": mean,
        "stdev": stdev,
        "sem": sem,
        "pooled_accept_rate": (tot_a / tot_d) if tot_d else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--trunk", default=DEFAULT_TRUNK,
                    help="trunk quant label, e.g. IQ2_M / Q5_K_S (resolves in the package)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: out/exp-049/drafter_quant_sweep[.<trunk>].json")
    ap.add_argument("--only", default=None,
                    help="comma-separated drafter labels to run (merge into existing --out)")
    args = ap.parse_args()

    trunk = PKG / f"gemma-4-31B-it-{args.trunk}.gguf"
    if not trunk.exists():
        print(f"error: trunk not found: {trunk}", file=sys.stderr)
        return 2
    if args.out is None:
        suffix = "" if args.trunk == DEFAULT_TRUNK else f".{args.trunk}"
        args.out = EXP / f"drafter_quant_sweep{suffix}.json"
    print(f"trunk: {args.trunk} ({trunk.name})  ->  {args.out}")

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    # Merge into existing results so --only adds a drafter without re-running others.
    results: dict[str, list[dict]] = {}
    if only and args.out.exists():
        results = json.loads(args.out.read_text())
    log_dir = args.out.parent / "logs"
    for dlabel, dpath in DRAFTERS.items():
        if only and dlabel not in only:
            continue
        if not dpath.exists():
            print(f"skip drafter {dlabel}: {dpath} missing", file=sys.stderr)
            continue
        results[dlabel] = []
        for n in N_VALUES:
            print(f"\n>>> drafter={dlabel}  n_max={n}  ({args.reps} reps)", flush=True)
            row = run_config(trunk, dpath, n, args.reps, args.max_tokens,
                             args.ctx, args.base_seed,
                             log_dir / f"{args.trunk}.{dlabel}.n{n}.log")
            m, sd = row["mean_accept_rate"], row["stdev"]
            if m is not None:
                print(f"    -> mean {m*100:.1f}% ± {sd*100:.1f}% (stdev)", flush=True)
            results[dlabel].append(row)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2))

    print(f"\nwrote {args.out}")
    print(f"\n=== {args.trunk} trunk: acceptance vs drafter quant (mean ± stdev %) ===")
    header = "| Drafter | " + " | ".join(f"n={n}" for n in N_VALUES) + " |"
    print(header)
    print("|" + "---|" * (len(N_VALUES) + 1))
    for dlabel, rows in results.items():
        cells = [f"{r['mean_accept_rate']*100:.1f}±{r['stdev']*100:.1f}%"
                 if r["mean_accept_rate"] is not None else "—" for r in rows]
        print(f"| {dlabel} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
