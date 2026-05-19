"""Run the tool-call benchmark N times per model with stochastic sampling and aggregate.

For each GGUF:
  1. Spawn llama-server once (saves ~30s × (N-1) reps per model of warmup).
  2. Run eval_toolcall.py via --base-url N times, each with a different seed.
  3. Tear down the server.

Per-rep rows go into toolcall_t0p7_results.csv. After all models complete, the
aggregator computes mean ± stdev across the N reps per model and writes
toolcall_t0p7_aggregated.csv (one row per model).

Wall-time estimate at default settings (n=25 holdout, 10 reps, 8 models): ~15 h.
"""

from __future__ import annotations

import csv
import math
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "out" / "omnicoder_q4_k_m"
LOGS = WORK / "logs"
HOLDOUT = WORK / "toolcall_holdout.jsonl"
RESULTS = WORK / "toolcall_reps_results.csv"
AGGREGATED = WORK / "toolcall_reps_aggregated.csv"
LLAMA_SERVER = REPO / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server"
EVAL_SCRIPT = REPO / "scripts" / "eval_toolcall.py"

# Sampling parameters tuned for tool-calling: lower temperature for stability,
# wider top_p (no nucleus pruning of tail), no presence penalty (penalties
# were suppressing valid tool calls in the prior attempt).
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}

REPETITIONS = 10
CTX = 16384
NGL = 99
SERVER_STARTUP_TIMEOUT = 120

MODELS = [
    WORK / "model-f16.gguf",
    WORK / "Q4_K_M-none.gguf",
    WORK / "Q4_K_M-stock_custom.gguf",
    WORK / "Q4_K_M-stock_wiki.gguf",
    WORK / "Q4_K_M-stock_mixed.gguf",
    WORK / "Q4_K_M-hybrid_custom.gguf",
    WORK / "Q4_K_M-hybrid_wiki.gguf",
    WORK / "Q4_K_M-hybrid_mixed.gguf",
]

PER_REP_METRICS = (
    "tool_selection_acc", "param_acc_mean", "schema_valid_rate",
    "rollout_complete_rate", "rollout_tool_set_match_rate",
    "n_turns",
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_health(base_url: str, timeout: float = SERVER_STARTUP_TIMEOUT) -> None:
    import requests
    url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise TimeoutError(f"llama-server at {url} not healthy in {timeout}s ({last_err})")


def spawn_server(model: Path, port: int) -> subprocess.Popen:
    log_path = LOGS / f"server_reps_{model.stem}_{int(time.time())}.log"
    log_fh = log_path.open("w")
    cmd = [
        str(LLAMA_SERVER),
        "-m", str(model),
        "--jinja",
        "--port", str(port),
        "-c", str(CTX),
        "-ngl", str(NGL),
        "--host", "127.0.0.1",
    ]
    print(f"  spawning server: {' '.join(cmd[:5])} … (log: {log_path.name})", flush=True)
    return subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)


def run_one_rep(model: Path, base_url: str, rep_index: int) -> None:
    """Run eval_toolcall.py once against a running server, with a per-rep seed."""
    seed = 1000 + rep_index  # deterministic but distinct per rep
    rep_log = LOGS / f"toolcall_t0p7_{model.stem}_rep{rep_index:02d}.log"
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--model", str(model),
        "--base-url", base_url,
        "--holdout", str(HOLDOUT),
        "--out", str(RESULTS),
        "--ctx", str(CTX),
        "--max-turns-per-session", "10",
        "--rollout-max-turns", "20",
        "--temperature", str(SAMPLING["temperature"]),
        "--top-p", str(SAMPLING["top_p"]),
        "--top-k", str(SAMPLING["top_k"]),
        "--min-p", str(SAMPLING["min_p"]),
        "--presence-penalty", str(SAMPLING["presence_penalty"]),
        "--repetition-penalty", str(SAMPLING["repetition_penalty"]),
        "--seed", str(seed),
    ]
    with rep_log.open("w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    return rc


def aggregate_per_model() -> None:
    """Compute mean ± stdev per metric across reps for each model."""
    if not RESULTS.exists():
        print(f"  no reps written to {RESULTS.name}; skipping aggregate", flush=True)
        return
    by_model: dict[str, list[dict]] = {}
    with RESULTS.open() as f:
        for row in csv.DictReader(f):
            by_model.setdefault(row["model"], []).append(row)

    out_fields = ["model", "n_reps"]
    for k in PER_REP_METRICS:
        out_fields += [f"{k}_mean", f"{k}_stdev"]
    out_fields += list(SAMPLING.keys())

    with AGGREGATED.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for model, rows in sorted(by_model.items()):
            agg: dict = {"model": model, "n_reps": len(rows)}
            for k in PER_REP_METRICS:
                vals = []
                for r in rows:
                    try:
                        vals.append(float(r[k]))
                    except (KeyError, ValueError):
                        pass
                if not vals:
                    agg[f"{k}_mean"] = ""
                    agg[f"{k}_stdev"] = ""
                    continue
                mean = sum(vals) / len(vals)
                stdev = (
                    math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
                    if len(vals) > 1 else 0.0
                )
                agg[f"{k}_mean"] = mean
                agg[f"{k}_stdev"] = stdev
            for k, v in SAMPLING.items():
                agg[k] = v
            w.writerow(agg)
    print(f"\nWrote aggregate: {AGGREGATED}", flush=True)


def main() -> int:
    assert HOLDOUT.exists(), HOLDOUT
    assert LLAMA_SERVER.exists(), LLAMA_SERVER
    LOGS.mkdir(parents=True, exist_ok=True)

    print(f"=== tool-call reps: {REPETITIONS} × {len(MODELS)} models ===", flush=True)
    print(f"sampling: {SAMPLING}", flush=True)
    print(f"holdout:  {HOLDOUT}", flush=True)
    print(f"results:  {RESULTS} (per-rep) / {AGGREGATED} (aggregated)", flush=True)

    t_total = time.time()
    for mi, model in enumerate(MODELS, 1):
        if not model.exists():
            print(f"[{mi}/{len(MODELS)}] SKIP missing {model.name}", flush=True)
            continue
        print(f"\n[{mi}/{len(MODELS)}] {model.name}", flush=True)
        port = free_port()
        proc = spawn_server(model, port)
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            wait_for_health(base_url)
            for rep in range(REPETITIONS):
                t0 = time.time()
                rc = run_one_rep(model, base_url, rep)
                dt = (time.time() - t0) / 60
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                print(f"    rep {rep + 1}/{REPETITIONS}: {status} ({dt:.1f} min)", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Incremental aggregate after each model so partial results are always usable.
        aggregate_per_model()

    aggregate_per_model()
    total_min = (time.time() - t_total) / 60
    print(f"\n=== ALL DONE in {total_min:.1f} min ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
