"""Tool-call smoke test for the exp-062 AWQ rungs vs the shipped ones.

Same shape as the release smoke we ran for the previous HF repo, on the Qwen3.8
holdout so the numbers line up with `out/exp-060-32k/eval/toolcall_smoke.csv`
(shipped IQ2_M: tool_sel 0.650 / param 0.306 / schema 1.000 over 92 scored turns).

Two things worth knowing before reading the output:

1. **The shipped rungs are re-measured here, in the same session, as controls.**
   The recorded 0.650 was taken on a different day against a different llama.cpp
   build. Comparing a fresh number to a stale one silently folds server drift into
   the delta; comparing two fresh ones does not.

2. **This is a tripwire, not the decision metric.** Greedy decoding over 92 scored
   turns puts the binomial SE near p=0.3 at ~4.8pp, so a swing under ~10pp is
   indistinguishable from noise. What the smoke *can* settle is the go/no-go
   question: does the quant still emit well-formed tool calls at all (schema
   validity, non-zero selection, no server crash). Rank the rungs with the full
   holdout eval afterwards, and decide on tool-call accuracy rather than KLD —
   the gemma-4-31B precedent lost on KLD and top_p and still won by +54% on tool
   arguments.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.eval.toolcall import Sampling, run_toolcall_eval  # noqa: E402

HOLDOUT = REPO / "out" / "exp-060-32k" / "eval" / "toolcall_holdout.jsonl"
LOG_DIR = REPO / "out" / "exp-062-32k" / "eval" / "logs-smoke"
OUT_CSV = REPO / "out" / "exp-062-32k" / "eval" / "toolcall_smoke.csv"

SEED = 42

# (label, path). Shipped rungs are controls; missing ones are skipped, not fatal —
# the IQ3_M rung was freed from disk after release and lives on the Hub.
MODELS = [
    ("IQ2_M-awq-new",
     REPO / "out/exp-062-awq-iq2m/gguf/IQ2_M-awq-best-hybrid_custom-mtp.gguf"),
    ("IQ2_M-shipped",
     REPO / "out/exp-060-32k/iq2_m/Qwen3.8-27B-IQ2_M.gguf"),
    ("IQ3_M-awq-new",
     REPO / "out/exp-062-awq-iq3m/gguf/IQ3_M-awq-best-hybrid_custom-mtp.gguf"),
    ("IQ3_M-shipped",
     REPO / "out/exp-060-32k/iq3_m/Qwen3.8-27B-IQ3_M.gguf"),
    ("IQ4_XS-awq-new",
     REPO / "out/exp-062-awq-iq4xs/gguf/IQ4_XS-awq-best-hybrid_custom-mtp.gguf"),
    ("IQ4_XS-shipped",
     REPO / "out/exp-060-32k/iq4_xs/Qwen3.8-27B-IQ4_XS.gguf"),
]


def main() -> int:
    if not HOLDOUT.exists():
        print(f"holdout missing: {HOLDOUT}", file=sys.stderr)
        return 1
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    sampling = Sampling(temperature=0.0, max_tokens=512, seed=SEED)
    rows: list[tuple[str, str | None, float, float, float, int]] = []

    present = [(lab, p) for lab, p in MODELS if p.exists()]
    missing = [lab for lab, p in MODELS if not p.exists()]
    if missing:
        print(f"[smoke] skipping (not on disk): {', '.join(missing)}")
    print(f"[smoke] {len(present)} models, holdout={HOLDOUT.name}, greedy\n")

    for i, (label, g) in enumerate(present, 1):
        print(f"=== [{i}/{len(present)}] {label} ===", flush=True)
        t0 = time.time()
        try:
            s = run_toolcall_eval(
                HOLDOUT, model_path=g,
                sampling=sampling,
                model_label=label,
                ctx=8192, ngl=99,
                server_log_path=LOG_DIR / f"{label}.server.log",
                per_turn_log=LOG_DIR / f"{label}.turns.jsonl",
                stop_on_fail=False,
            )
            dt = time.time() - t0
            print(f"  tool_sel={s.tool_selection_acc:.3f}  "
                  f"param={s.param_acc_mean:.3f}  "
                  f"schema={s.schema_valid_rate:.3f}  "
                  f"n_scored={s.n_scored}  ({dt:.0f}s)\n", flush=True)
            rows.append((label, None, s.tool_selection_acc, s.param_acc_mean,
                         s.schema_valid_rate, s.n_scored))
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            print(f"  FAILED after {dt:.0f}s: {type(e).__name__}: {e}\n", flush=True)
            rows.append((label, f"{type(e).__name__}: {e}", 0.0, 0.0, 0.0, 0))

    print("=== SMOKE SUMMARY (greedy; SE ~4.8pp at n=92 — read as a tripwire) ===\n")
    print(f"  {'model':22s}  tool_sel   param  schema    n  notes")
    print(f"  {'-'*22}  --------  ------  ------  ---  -----")
    for label, err, sel, param, schema, n in rows:
        note = "OK" if err is None else f"ERR {err[:40]}"
        print(f"  {label:22s}  {sel:8.3f}  {param:6.3f}  {schema:6.3f}  {n:>3d}  {note}")

    # Paired deltas, new minus its own shipped control.
    by = {lab: (err, sel, param, schema) for lab, err, sel, param, schema, _n in rows}
    print()
    for rung in ("IQ2_M", "IQ3_M", "IQ4_XS"):
        new, old = by.get(f"{rung}-awq-new"), by.get(f"{rung}-shipped")
        if new and old and new[0] is None and old[0] is None:
            print(f"  {rung}: tool_sel {new[1]-old[1]:+.3f}   "
                  f"param {new[2]-old[2]:+.3f}   schema {new[3]-old[3]:+.3f}"
                  "   (new - shipped)")
        else:
            print(f"  {rung}: no paired comparison (a side is missing or errored)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_CSV.exists()
    with OUT_CSV.open("a") as f:
        if write_header:
            f.write("model,tool_selection_acc,param_acc_mean,schema_valid_rate,"
                    "n_scored,error\n")
        for label, err, sel, param, schema, n in rows:
            f.write(f"{label},{sel},{param},{schema},{n},{err or ''}\n")
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
