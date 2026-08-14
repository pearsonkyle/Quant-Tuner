#!/usr/bin/env python
"""Measure what a (window, trained_tail) pair actually costs, in the real training loop.

The last sizing decision was made from a single-window forward/backward probe. It omitted
the optimizer step and ran before macOS had built any swap, and it picked a window that
could not complete a step at all. So this does not probe: it writes a synthetic corpus at
the requested window and runs `qat.train` over it for a few real steps — same model, same
wrapping, same adafactor + fp32 masters, same grad-accum, same checkpointing — and reports
s/step and the swap delta alongside the allocator number.

Synthetic token ids are fine here: memory and time depend on shape and label density, not
on what the tokens mean. `--labeled-frac` should match the corpus you intend to train on
(the SFT blob's density floor is 0.05; the mean is ~0.35).

    python scripts/probe_window_budget.py --window 32768 --trained-tail 8192
    python scripts/probe_window_budget.py --grid          # the whole sweep, one process each

Read `s/step` against `swap_delta_gib`: a step that is slow with flat swap is compute-bound
and merely expensive; a step that is slow with swap climbing is the failure mode that
killed the 16128 attempt, and it does not improve with patience.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: (window, trained_tail) — 0 tail means full-gradient, the configuration that failed above
#: 12288. The paired rows are what show whether the prefix split is what bought the window.
#: A LARGER tail is better per trained token, not smaller: attention cost is ~S^2 per
#: window regardless of the split, so cost per trained token goes as S^2/T. Minimizing the
#: tail minimizes memory and maximizes cost. Sweep upward until it stops fitting.
GRID = [(8064, 0), (32768, 8192), (32768, 16384), (32768, 24576)]


def swap_used_gib() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used\s*=\s*([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


def write_corpus(path: Path, window: int, n_win: int, labeled_frac: float) -> None:
    import torch
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, 150_000, (n_win, window), generator=g)
    lbl = ids.clone()
    # Label a contiguous TAIL block per window, not a random scatter: that is the shape a
    # trajectory corpus has, and it is what decides how many labeled positions survive a
    # prefix split (a scatter would flatter the prefix path).
    keep = max(1, int(window * labeled_frac))
    lbl[:, :window - keep] = -100
    torch.save({"ids": ids, "labels": lbl, "window": window,
                "im_end_id": 151645, "im_end_targets": n_win,
                "fingerprint": f"probe-{window}-{labeled_frac}",
                "assistant_frac": labeled_frac}, path)


def run_one(window: int, tail: int, *, steps: int, accum: int, labeled_frac: float,
            model_dir: Path | None, extra: list[str], timeout: float) -> dict:
    tmp = REPO / "out" / "probe" / f"corpus-{window}-{labeled_frac}.pt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    write_corpus(tmp, window, steps * accum, labeled_frac)

    cmd = [sys.executable, "-m", "quant_tuner.qat.train",
           "--corpus", str(tmp), "--train-layers", "36", "--optim", "adafactor",
           "--dtype", "fp32", "--grad-accum", str(accum), "--epochs", "1.0",
           "--lr", "5e-4", "--ckpt-every", "0", "--val-every", "0",
           "--out", str(REPO / "out" / "probe" / "trained"), *extra]
    if tail:
        cmd += ["--trained-tail", str(tail)]
    if model_dir:
        cmd += ["--model-dir", str(model_dir)]

    swap0 = swap_used_gib()
    t0 = time.time()
    timed_out = False
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        log, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        # A swap-bound configuration does not fail, it just never finishes — the 16128
        # attempt ran 31 minutes without completing one step. Cap it and record that.
        timed_out = True
        log = (e.stdout or "") + (e.stderr or "")
        if isinstance(log, bytes):
            log = log.decode(errors="replace")
        rc = -1
    wall = time.time() - t0

    s_step = [float(m) for m in re.findall(r"([\d.]+)s/step", log)]
    mem = [float(m) for m in re.findall(r"mem=([\d.]+)GiB", log)]
    # A macOS OOM kill leaves no traceback — the process just dies. Distinguish that from
    # a python error, because the two mean opposite things about the configuration.
    killed = rc != 0 and not timed_out and "Traceback" not in log
    return {"window": window, "trained_tail": tail, "steps_logged": len(s_step),
            "s_per_step": s_step[-1] if s_step else None,
            "mps_gib": max(mem) if mem else None,
            "swap_delta_gib": round(swap_used_gib() - swap0, 2),
            "wall_s": round(wall, 1), "returncode": rc,
            "timed_out": timed_out, "killed_no_traceback": killed,
            "error": None if rc == 0 else (log.strip().splitlines()[-1:] or ["no output"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int)
    ap.add_argument("--trained-tail", type=int, default=0)
    ap.add_argument("--grid", action="store_true", help=f"sweep {GRID}")
    ap.add_argument("--steps", type=int, default=3,
                    help="real optimizer steps to complete (default 3; the first is "
                         "warm-up-dominated, so read the last)")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--labeled-frac", type=float, default=0.35)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--cooldown", type=float, default=120,
                    help="seconds to wait between configurations (default 120). Without "
                         "it a config starts while macOS is still holding the previous "
                         "one's swap and gets SIGKILLed during model load — which reads "
                         "as 'this configuration OOMs' when it never ran.")
    ap.add_argument("--timeout", type=float, default=2400,
                    help="seconds per configuration (default 2400). A swap-bound config "
                         "does not error, it just never completes a step.")
    ap.add_argument("--out", type=Path, default=REPO / "out" / "probe" / "budget.json")
    args, extra = ap.parse_known_args()

    combos = GRID if args.grid else [(args.window, args.trained_tail)]
    if not args.grid and not args.window:
        return ap.error("pass --window or --grid")

    rows = []
    for n, (window, tail) in enumerate(combos):
        if n and args.cooldown:
            print(f"[cooldown] {args.cooldown:.0f}s for the previous run's swap to drain "
                  f"(now {swap_used_gib():.1f} GiB)", flush=True)
            time.sleep(args.cooldown)
        print(f"\n=== window {window}  trained_tail {tail or 'full'} ===", flush=True)
        r = run_one(window, tail, steps=args.steps, accum=args.grad_accum,
                    labeled_frac=args.labeled_frac, model_dir=args.model_dir,
                    extra=extra, timeout=args.timeout)
        rows.append(r)
        print(json.dumps(r), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))

    print(f"\n{'window':>8} {'tail':>8} {'s/step':>9} {'MPS GiB':>9} {'swap Δ':>8}  status")
    for r in rows:
        status = ("ok" if r["returncode"] == 0 else
                  f"no step in {args.timeout/60:.0f} min (swap-bound)" if r["timed_out"] else
                  "OOM-killed (no traceback)" if r["killed_no_traceback"] else "error")
        print(f"{r['window']:>8} {r['trained_tail'] or 'full':>8} "
              f"{r['s_per_step'] or float('nan'):>9.1f} "
              f"{r['mps_gib'] or float('nan'):>9.1f} {r['swap_delta_gib']:>8.2f}  {status}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
