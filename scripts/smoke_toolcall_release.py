"""Smoke-test tool-calling on every shipped GGUF in the upload dir.

Picks 5 random sessions from the tool-call holdout (seed=42 for reproducibility)
and runs each against every *.gguf in the release directory. Per-model server
spin-up + a handful of tool-call turns; ~2 min per model, ~25 min total for all
12 files.

Outputs a compact comparison table:
    file                                            tool_sel  param  schema  n_scored
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.eval.toolcall import Sampling, run_toolcall_eval

HOLDOUT = REPO / "out" / "exp-002" / "toolcall_holdout.jsonl"
UPLOAD = REPO / "uploads" / "pearsonkyle" / "gemma-4-31B-it-awq-2bit-GGUF"
LOG_DIR = REPO / "out" / "smoke" / "release_verify"

N_SAMPLES = 5
SEED = 42


def _sample_holdout(n: int, seed: int) -> Path:
    lines = HOLDOUT.read_text().splitlines()
    rng = random.Random(seed)
    picks = rng.sample(lines, n)
    tmp = Path(tempfile.mkstemp(suffix=".jsonl", prefix="toolcall_smoke_")[1])
    tmp.write_text("\n".join(picks) + "\n")
    print(f"[smoke] sampled {n} sessions seed={seed} -> {tmp}")
    for i, s in enumerate(picks, 1):
        d = json.loads(s)
        print(f"  {i}. {d.get('session_id', '?')[:40]:40s}  "
              f"tool_calls={d.get('tool_call_count', '?')}")
    return tmp


def main() -> int:
    if not HOLDOUT.exists():
        print(f"holdout missing: {HOLDOUT}", file=sys.stderr)
        return 1
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    holdout = _sample_holdout(N_SAMPLES, SEED)
    ggufs = sorted(UPLOAD.glob("*.gguf"))
    print(f"\n[smoke] {len(ggufs)} GGUFs to test\n")

    sampling = Sampling(temperature=0.0, max_tokens=512, seed=SEED)
    rows: list[tuple[str, str | None, float, float, float, int]] = []

    for i, g in enumerate(ggufs, 1):
        label = g.name.replace("gemma-4-31B-it-", "").replace(".gguf", "")
        print(f"\n=== [{i}/{len(ggufs)}] {label} ===", flush=True)
        t0 = time.time()
        try:
            summary = run_toolcall_eval(
                holdout, model_path=g,
                sampling=sampling,
                model_label=label,
                ctx=8192, ngl=99,
                server_log_path=LOG_DIR / f"{label}.server.log",
                per_turn_log=LOG_DIR / f"{label}.turns.jsonl",
                stop_on_fail=False,
            )
            dt = time.time() - t0
            print(f"  tool_sel={summary.tool_selection_acc:.3f}  "
                  f"param={summary.param_acc_mean:.3f}  "
                  f"schema={summary.schema_valid_rate:.3f}  "
                  f"n_scored={summary.n_scored}  ({dt:.0f}s)")
            rows.append((
                label, None,
                summary.tool_selection_acc,
                summary.param_acc_mean,
                summary.schema_valid_rate,
                summary.n_scored,
            ))
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            print(f"  FAILED after {dt:.0f}s: {type(e).__name__}: {e}",
                  flush=True)
            rows.append((label, f"{type(e).__name__}: {e}",
                          0.0, 0.0, 0.0, 0))

    print("\n=== SMOKE TEST SUMMARY ===")
    print(f"  holdout sample: {N_SAMPLES} sessions, seed={SEED}\n")
    print(f"  {'file':50s}  tool_sel   param   schema   n  notes")
    print(f"  {'-'*50}  --------  ------  ------  ---  -----")
    for label, err, tool_sel, param, schema, n in rows:
        note = "OK" if err is None else f"ERR {err[:30]}"
        print(f"  {label:50s}  {tool_sel:8.3f}  {param:6.3f}  "
              f"{schema:6.3f}  {n:>3d}  {note}")

    holdout.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
