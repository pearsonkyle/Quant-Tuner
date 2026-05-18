"""Run the tool-call benchmark on every GGUF in the workspace, in series.

The eval script spawns llama-server per model so we can't parallelize; each
run lasts ~5-15 min on a 9B Q4_K_M. Idempotent: skips any (model, holdout)
combination that already has a row in toolcall_results.csv with a recent
timestamp.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "out" / "omnicoder_q4_k_m"
LOGS = WORK / "logs"
HOLDOUT = WORK / "toolcall_holdout.jsonl"
RESULTS = WORK / "toolcall_results.csv"

MODELS = [
    WORK / "model-f16.gguf",
    WORK / "Q4_K_M-none.gguf",
    WORK / "Q4_K_M-stock_custom.gguf",
    WORK / "Q4_K_M-stock_wiki.gguf",
    WORK / "Q4_K_M-hybrid_custom.gguf",
    WORK / "Q4_K_M-hybrid_wiki.gguf",
]


def already_run(model: Path) -> bool:
    if not RESULTS.exists():
        return False
    name = model.name
    with RESULTS.open() as f:
        for row in csv.DictReader(f):
            if row.get("model") == name:
                return True
    return False


def main() -> int:
    assert HOLDOUT.exists(), f"missing holdout: {HOLDOUT}"
    LOGS.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        if not m.exists():
            print(f"[skip] missing {m.name}", flush=True)
            continue
        if already_run(m):
            print(f"[skip] {m.name} already in {RESULTS.name}", flush=True)
            continue
        print(f"\n==> eval {m.name}", flush=True)
        t0 = time.time()
        log_path = LOGS / f"toolcall_{m.stem}.log"
        cmd = [
            sys.executable, str(REPO / "scripts" / "eval_toolcall.py"),
            "--model", str(m),
            "--holdout", str(HOLDOUT),
            "--out", str(RESULTS),
            "--ctx", "16384",
            "--max-turns-per-session", "10",
            "--rollout-max-turns", "20",
        ]
        with log_path.open("w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        dt = (time.time() - t0) / 60
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"    {status}  ({dt:.1f} min)  log: {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
