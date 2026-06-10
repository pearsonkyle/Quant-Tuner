#!/usr/bin/env python3
"""Denoising-throughput bench for diffusion GGUFs (DiffusionGemma et al.).

llama-bench measures autoregressive prefill/decode — decode tok/s is NOT how
a block-diffusion model generates. This script times `llama-diffusion-cli`
end-to-end instead: wall time per generated token across N repetitions
(mean ± stdev), the diffusion analog of bench/speed.py.

Usage:
  uv run python scripts/bench_diffusion_speed.py \
      --model out/run/gguf/IQ2_M-awq-hybrid_custom.gguf \
      --prompt "Explain the Fourier transform." \
      --n-predict 256 --reps 5 --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_tuner.paths import llama_bin  # noqa: E402

# entropy-bound CLI timing line: "total time: 1234.56ms, time per step: ..."
_TOTAL_RE = re.compile(r"total time:\s*([0-9.]+)ms.*\((\d+) steps over (\d+) blocks", re.S)


def run_once(args, rep: int) -> dict:
    cmd = [
        str(llama_bin("llama-diffusion-cli")),
        "-m", str(args.model),
        "-p", args.prompt,
        "-n", str(args.n_predict),
        "-ngl", str(args.ngl),
        "--seed", str(args.seed + rep),
    ]
    if args.diffusion_steps:
        cmd += ["--diffusion-steps", str(args.diffusion_steps)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    wall_s = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"llama-diffusion-cli failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    combined = proc.stdout + proc.stderr
    row = {"rep": rep, "wall_s": round(wall_s, 3)}
    m = _TOTAL_RE.search(combined)
    if m:
        row["gen_ms"] = float(m.group(1))
        row["steps"] = int(m.group(2))
        row["blocks"] = int(m.group(3))
        row["ms_per_step"] = round(row["gen_ms"] / max(row["steps"], 1), 2)
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prompt", default="Write a Python function that merges two sorted lists.")
    p.add_argument("--n-predict", type=int, default=256)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--diffusion-steps", type=int, default=0,
                   help="override step count (0 = model/GGUF default)")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", type=Path, default=None, help="CSV output path")
    args = p.parse_args()

    rows = []
    for rep in range(args.reps):
        row = run_once(args, rep)
        rows.append(row)
        print(f"rep {rep}: {row}", flush=True)

    walls = [r["wall_s"] for r in rows]
    mean = statistics.mean(walls)
    stdev = statistics.stdev(walls) if len(walls) > 1 else 0.0
    tok_s = args.n_predict / mean if mean > 0 else 0.0
    print(f"\n{args.model.name}: wall {mean:.2f}s ± {stdev:.2f} "
          f"(~{tok_s:.1f} tok/s incl. model load, n={args.reps})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cols = sorted({k for r in rows for k in r})
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", *cols])
            w.writeheader()
            for r in rows:
                w.writerow({"model": args.model.name, **r})
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
