#!/usr/bin/env python
"""Periodic markdown progress report for a running (or finished) QAT job.

`watch_qat_run.sh` answers "is it still alive and is swap climbing" on macOS. This answers
the different question — *is it learning, and is the data doing what we hoped* — from
`metrics.jsonl` alone, so it is device-agnostic: no MPS calls, no `sysctl`, nothing that
assumes the trainer is local or still running.

Four things it surfaces that a loss curve cannot:

* **Code flips.** A ternary model only learns by flipping codes; the loss falls on scale
  drift alone. `flip_pct_delta` is the velocity — a run whose velocity has peaked on every
  tensor is converging, one flipping ~0% is not training regardless of the loss.
* **Loss by source.** The trainer records per-source loss because the corpus mixes sources
  with very different assistant fractions (0.08 refusals .. 0.79 broad-instruct). A single
  curve cannot say which data is driving the flips; this table can, and it is the evidence
  for a mixture change on the next run.
* **Validation cost.** `val_seconds` against `s_per_step` — at a long window validation
  can silently cost more than the steps between it.
* **The grad-spike guard.** `n_skipped` climbing means windows are being dropped.

    python scripts/qat_progress_report.py out/exp-058/run             # once, to stdout
    python scripts/qat_progress_report.py out/exp-058/run --watch 1800   # every 30 min
    python scripts/qat_progress_report.py out/exp-058/run --out report.md

With `--watch` the report is rewritten in place each interval (and appended to
`<out>/report_history.md`), so checking in means reading a file rather than re-polling a
process that has nothing new to say for the next 20 minutes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load(metrics: Path) -> tuple[list[dict], list[dict], dict[int, dict[str, dict]]]:
    """-> (step rows, val rows, {step: {tensor: flip stats}}). Tolerates a partial last
    line: the trainer flushes per record, but a report can still land mid-write."""
    steps: list[dict] = []
    vals: list[dict] = []
    flips: dict[int, dict[str, dict]] = {}
    if not metrics.exists():
        return steps, vals, flips
    for line in metrics.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = d.get("kind")
        if kind == "step":
            steps.append(d)
        elif kind == "val":
            vals.append(d)
        elif kind == "flip":
            flips.setdefault(d["step"], {})[d.get("tensor", "?")] = d
    return steps, vals, flips


def hms(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def trend(values: list[float], n: int = 5) -> str:
    """Compare the mean of the first n against the last n. Reported as a direction, not a
    verdict — on a ternary run a falling loss is not by itself evidence of learning."""
    if len(values) < 2:
        return "n/a"
    k = min(n, len(values) // 2) or 1
    a = sum(values[:k]) / k
    b = sum(values[-k:]) / k
    return f"{a:.4f} -> {b:.4f} ({b - a:+.4f})"


def render(out_dir: Path) -> str:
    steps, vals, flips = load(out_dir / "metrics.jsonl")
    L = [f"# QAT progress — `{out_dir}`", ""]
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.")
    L.append("")

    if not steps:
        L.append("No `step` records yet in `metrics.jsonl`.")
        L.append("")
        L.append("At a long window the first step can take 15-30 min, and the trainer only")
        L.append("emits at step 1 then every 5th step. If the file is missing entirely,")
        L.append("check that the process started and that `--metrics-jsonl` is on.")
        return "\n".join(L)

    last = steps[-1]
    total = last.get("total_steps") or 0
    done = last["step"]
    sps = last.get("s_per_step") or 0.0
    frac = done / total if total else 0.0
    eta = (total - done) * sps

    L += ["## Where it is", ""]
    L.append(f"- step **{done}/{total}** ({frac * 100:.1f}%)")
    L.append(f"- elapsed **{hms(last.get('elapsed_s', 0))}**, ETA **{hms(eta)}** at "
             f"{sps:.0f} s/step")
    L.append(f"- tokens seen **{last.get('tokens_seen', 0):,}**"
             + (f" ({last['tokens_seen'] / max(1e-9, last.get('elapsed_s', 1)):.0f} tok/s)"
                if last.get("tokens_seen") else ""))
    if last.get("mem_gib"):
        L.append(f"- device memory **{last['mem_gib']:.1f} GiB** "
                 f"(0 means the trainer has no reporter for this device)")
    L.append(f"- lr {last.get('lr', 0):.2e}")
    L.append("")

    L += ["## Loss", "",
          f"- overall: {trend([s['loss'] for s in steps])}",
          ""]
    by_src: dict[str, list[float]] = {}
    for s in steps:
        for k, v in (s.get("loss_by_source") or {}).items():
            by_src.setdefault(k, []).append(v)
    if len(by_src) > 1:
        L += ["Per source — **this is the mixture evidence**. A source whose loss is flat "
              "while others fall is either already fit or not being learned; either way it "
              "is a candidate for reweighting on the next run.", "",
              "| source | windows scored | first -> last |", "|---|---|---|"]
        for k in sorted(by_src, key=lambda k: -len(by_src[k])):
            L.append(f"| `{k}` | {len(by_src[k])} | {trend(by_src[k])} |")
        L.append("")
    elif by_src:
        L.append(f"Single source in this corpus: `{next(iter(by_src))}`.")
        L.append("")

    L += ["## Code flips — read this, not the loss", ""]
    if not flips:
        L.append("No `flip` records yet — they are written at each checkpoint, so nothing "
                 "appears until the first `--ckpt-every` boundary.")
        L.append("")
    else:
        at = max(flips)
        rows = flips[at]
        L.append(f"At step **{at}**, versus the start-of-run snapshot:")
        L.append("")
        L += ["| tensor | flips % | Δ since last | 0->± | ±->0 | density | scale drift |",
              "|---|---|---|---|---|---|---|"]
        for name, st in sorted(rows.items()):
            d = st.get("flip_pct_delta")
            L.append(
                f"| `{name}` | {st.get('flip_pct', 0):.4f} | "
                f"{f'{d:+.4f}' if d is not None else '—'} | "
                f"{st.get('zero_to_nonzero', 0):,} | {st.get('nonzero_to_zero', 0):,} | "
                f"{st.get('density_start', 0) * 100:.1f} -> {st.get('density', 0) * 100:.1f}% | "
                f"{st.get('scale_drift', 0) * 100:.2f}% "
                f"({st.get('scale_drift_signed', 0) * 100:+.2f}%) |")
        L.append("")
        total_flip = sum(s.get("flip_pct", 0) for s in rows.values()) / max(1, len(rows))
        if total_flip < 0.01:
            L.append("> **Mean flips under 0.01% — the model is not learning.** Scale drift "
                     "alone will still pull the loss down. This is the lr-3e-4 failure "
                     "mode; raise the LR rather than waiting.")
        else:
            L.append(f"> Mean flips {total_flip:.4f}% across {len(rows)} sampled tensors. "
                     "Judge by the Δ column: still rising = learning, peaked on every "
                     "tensor = converging, zero = stalled.")
        L.append("")

    L += ["## Gradient health", ""]
    L.append(f"- grad norm {last.get('grad_norm', 0):.3f} (guard median "
             f"{last.get('grad_median', 0):.3f})")
    skipped = last.get("n_skipped", 0)
    L.append(f"- windows skipped by the spike guard: **{skipped}**"
             + (" — worth checking which ones" if skipped else ""))
    if last.get("n_tail_empty"):
        L.append(f"- windows whose trained tail had no labels: {last['n_tail_empty']} "
                 "(pure waste; shorten the prefix or drop `--trained-tail`)")
    L.append("")

    L += ["## Validation", ""]
    if not vals:
        L.append("No `val` records yet (`--val-every`/`--val-corpus` may be off).")
    else:
        v = vals[-1]
        L.append(f"- last: masked-CE **{v['val_masked_ce']:.4f}** at step {v['step']} "
                 f"over {v.get('val_windows', '?')} windows")
        L.append(f"- curve: {trend([x['val_masked_ce'] for x in vals])}")
        cost = v.get("val_seconds")
        if cost:
            share = cost / max(1e-9, sps)
            L.append(f"- cost **{cost:.0f}s**, i.e. {share:.2f} steps' worth of wall-clock")
            if share > 1.0:
                L.append("> Validation costs more than a training step. Lower "
                         "`--val-windows` or raise `--val-every`.")
    L.append("")

    L += ["## Recent steps", "", "| step | loss | grad norm | s/step | elapsed |",
          "|---|---|---|---|---|"]
    for s in steps[-10:]:
        L.append(f"| {s['step']} | {s['loss']:.4f} | {s.get('grad_norm', 0):.2f} | "
                 f"{s.get('s_per_step', 0):.0f} | {hms(s.get('elapsed_s', 0))} |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path, help="the trainer's --out directory")
    ap.add_argument("--out", type=Path,
                    help="write here (default <out_dir>/report.md; '-' for stdout only)")
    ap.add_argument("--watch", type=float, metavar="SECONDS",
                    help="regenerate on this interval instead of once. 1800 (30 min) is "
                         "a sane floor at a long window — a step is 15-30 min.")
    args = ap.parse_args()

    dest = None if str(args.out) == "-" else (args.out or args.out_dir / "report.md")
    while True:
        text = render(args.out_dir)
        print(text, flush=True)
        if dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            # Keep the series too: the flip VELOCITY is only readable across reports.
            with (args.out_dir / "report_history.md").open("a") as fh:
                fh.write(text + "\n\n---\n\n")
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
