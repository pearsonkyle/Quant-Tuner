"""exp-048: MTP draft-acceptance sweep WITH error bars (reps) for gemma-4-31B.

Successor to exp-046/exp-047. Same fixed 5-prompt set and the root
`mtp-gemma-4-31B-it.gguf` drafter, but runs every (quant, n_max) config for
`--reps` independent reps (distinct seeds) so we can report a mean ± stdev
acceptance rate instead of a single pooled point estimate.

For each (quant, n_max) it spawns ONE `llama-server` and runs `reps` reps over
the 5 prompts; one rep = the 5 prompts pooled (sum accepted / sum drafted from
`timings.draft_n` / `timings.draft_n_accepted`). The acceptance rate per config
is the mean over reps; the error bar is the sample stdev (and sem) across reps.

Usage:
    .venv/bin/python scripts/exp048_mtp_acceptance_reps.py \
        --pkg uploads/pearsonkyle/gemma-4-31b-it-imatrix-GGUF \
        --reps 3 --max-tokens 200 --out out/exp-048/acceptance_reps.json
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

# Trunk quants in the package (label -> filename). Q5_K_S included (exp-047).
QUANTS = {
    "Q5_K_S": "gemma-4-31B-it-Q5_K_S.gguf",
    "IQ4_XS": "gemma-4-31B-it-IQ4_XS.gguf",
    "IQ3_M": "gemma-4-31B-it-IQ3_M.gguf",
    "IQ2_M": "gemma-4-31B-it-IQ2_M.gguf",
}
DRAFTER_FILE = "mtp-gemma-4-31B-it.gguf"  # root drafter (same as exp-047)
N_VALUES = [1, 2, 3, 4]

# Mixed coding + reasoning prompts (identical to exp-046/047).
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
    """Spawn one server; run `reps` reps over the prompt set. One rep = pooled
    accept rate over the 5 prompts. Returns mean/stdev/sem across reps."""
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
            # warmup (not recorded)
            requests.post(base, json={"messages": [{"role": "user", "content": "Say hi."}],
                                      "max_tokens": 8}, timeout=120)
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
    ap.add_argument("--pkg", type=Path,
                    default=REPO / "uploads/pearsonkyle/gemma-4-31b-it-imatrix-GGUF")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=REPO / "out/exp-048/acceptance_reps.json")
    args = ap.parse_args()

    drafter = args.pkg / DRAFTER_FILE
    if not drafter.exists():
        print(f"error: drafter not found: {drafter}", file=sys.stderr)
        return 2

    results: dict[str, list[dict]] = {}
    log_dir = args.out.parent / "logs"
    for qlabel, qfile in QUANTS.items():
        trunk = args.pkg / qfile
        if not trunk.exists():
            print(f"skip {qlabel}: {trunk} missing", file=sys.stderr)
            continue
        results[qlabel] = []
        for n in N_VALUES:
            print(f"\n>>> {qlabel}  n_max={n}  ({args.reps} reps)", flush=True)
            row = run_config(trunk, drafter, n, args.reps, args.max_tokens,
                             args.ctx, args.base_seed, log_dir / f"{qlabel}.n{n}.log")
            m, sd = row["mean_accept_rate"], row["stdev"]
            if m is not None:
                print(f"    -> mean {m*100:.1f}% ± {sd*100:.1f}% (stdev)", flush=True)
            results[qlabel].append(row)
            # incremental save so a crash mid-sweep keeps finished configs
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2))

    print(f"\nwrote {args.out}")

    print("\n=== mean acceptance rate ± stdev (%) ===")
    header = "| Quant | " + " | ".join(f"n={n}" for n in N_VALUES) + " |"
    print(header)
    print("|" + "---|" * (len(N_VALUES) + 1))
    for qlabel, rows in results.items():
        cells = []
        for r in rows:
            if r["mean_accept_rate"] is not None:
                cells.append(f"{r['mean_accept_rate']*100:.1f}±{r['stdev']*100:.1f}%")
            else:
                cells.append("—")
        print(f"| {qlabel} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
