"""exp-046: MTP draft-acceptance sweep for the gemma-4-31B imatrix release.

Pairs each shipped trunk quant (IQ2_M / IQ3_M / IQ4_XS) with the converted
`google/gemma-4-31B-it-assistant` drafter (`MTP/gemma-4-31B-it-Q8_0-MTP.gguf`,
arch `gemma4-assistant`, nextn_predict_layers=4) and measures llama.cpp draft
acceptance at `--spec-draft-n-max` = 1, 2, 3, 4.

For each (quant, n) it spawns one `llama-server` with
`--model-draft <drafter> --spec-type draft-mtp --spec-draft-n-max n`, fires a
fixed prompt set, and reads the per-response `timings.draft_n` /
`timings.draft_n_accepted` to compute the overall acceptance rate. Mean
acceptance length (accepted tokens per drafted block) is also reported. Speed is
deliberately ignored — this machine's thermals make tok/s non-comparable.

Usage:
    .venv/bin/python scripts/exp046_mtp_acceptance.py \
        --pkg uploads/pearsonkyle/gemma-4-31b-it-imatrix-GGUF \
        --reps 5 --max-tokens 200 --out out/exp-046/acceptance.json
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
LLAMA_SERVER = REPO / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server"

# Trunk quants in the package (label -> filename).
QUANTS = {
    "IQ4_XS": "gemma-4-31B-it-IQ4_XS.gguf",
    "IQ3_M": "gemma-4-31B-it-IQ3_M.gguf",
    "IQ2_M": "gemma-4-31B-it-IQ2_M.gguf",
}
DRAFTER = "MTP/gemma-4-31B-it-Q8_0-MTP.gguf"
N_VALUES = [1, 2, 3, 4]

# Mixed coding + reasoning prompts (this is a coder/tool-use model).
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
               ctx: int, log_path: Path) -> dict:
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
    drafted = accepted = blocks = 0
    with log_path.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        try:
            wait_health(port, proc)
            base = f"http://127.0.0.1:{port}/v1/chat/completions"
            # warmup
            requests.post(base, json={"messages": [{"role": "user", "content": "Say hi."}],
                                      "max_tokens": 8}, timeout=120)
            for prompt in PROMPTS[:reps]:
                r = requests.post(base, json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens, "temperature": 0.3,
                    "chat_template_kwargs": {"enable_thinking": False},
                }, timeout=600)
                r.raise_for_status()
                t = r.json().get("timings", {}) or {}
                d = t.get("draft_n", 0) or 0
                a = t.get("draft_n_accepted", 0) or 0
                drafted += d
                accepted += a
                if d:
                    # one drafted block per n tokens (approx) — count blocks by draft calls
                    blocks += max(1, round(d / n))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    accept_rate = (accepted / drafted) if drafted else None
    # mean accepted tokens per drafted block = accepted / blocks
    mean_acc_len = (accepted / blocks) if blocks else None
    return {
        "n_max": n,
        "draft_total": drafted,
        "draft_accepted": accepted,
        "accept_rate": accept_rate,
        "mean_accept_len": mean_acc_len,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkg", type=Path,
                    default=REPO / "uploads/pearsonkyle/gemma-4-31b-it-imatrix-GGUF")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=REPO / "out/exp-046/acceptance.json")
    args = ap.parse_args()

    drafter = args.pkg / DRAFTER
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
            print(f"\n>>> {qlabel}  n_max={n}", flush=True)
            row = run_config(trunk, drafter, n, args.reps, args.max_tokens,
                             args.ctx, log_dir / f"{qlabel}.n{n}.log")
            ar = row["accept_rate"]
            mal = row["mean_accept_len"]
            print(f"    accept_rate={ar*100:.1f}%  ({row['draft_accepted']}/{row['draft_total']})  "
                  f"mean_accept_len={mal:.2f}" if ar is not None else "    no draft stats")
            results[qlabel].append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")

    # markdown table: accept rate vs n vs quant
    print("\n=== acceptance rate (%) ===")
    header = "| Quant | " + " | ".join(f"n={n}" for n in N_VALUES) + " |"
    print(header)
    print("|" + "---|" * (len(N_VALUES) + 1))
    for qlabel, rows in results.items():
        cells = []
        for r in rows:
            cells.append(f"{r['accept_rate']*100:.1f}%" if r["accept_rate"] is not None else "—")
        print(f"| {qlabel} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
