#!/usr/bin/env python3
"""Pick the curriculum's --stop-weight from the ablation's measured termination behaviour.

    STOP_WEIGHT=$(python scripts/choose_stop_weight.py --tag sft32k_sw1)

Prints ONE number on stdout (so a shell can capture it) and the reasoning on stderr.

There are two opposite termination failures and one knob between them, so the choice
cannot be made from a single "is it good" score:

  * **never stops** — the model loops. Measured as a LOW P(stop) at `after_tool_call`,
    or behaviourally as a repeated command in the agent trajectory. The shipped vanilla
    Q2_0 does this: 19 tool calls, 4 distinct commands. Fix: raise the stop weight.
  * **stops too early** — the model emits its stop token after the first sentence and
    never gets to the task. Measured as a HIGH P(stop) at `sentence_period`, or
    behaviourally as an episode with ~0 output tokens. `--stop-weight 6.0` produced this
    (P=0.974 against vanilla's 0.009). Fix: lower the stop weight.

So the probe alone is ambiguous in one direction and the trajectory alone is ambiguous in
the other; this reads both, and refuses to guess when they disagree.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STOP_CSV = REPO / "out" / "exp-058" / "eval" / "stop_prob.csv"

DIAG = "sentence_period"      # high here = stops too early
CTRL = "after_tool_call"      # low here = never stops

# Vanilla's measured values, the healthy reference for this base model.
VANILLA_DIAG = 0.0092
VANILLA_CTRL = 0.99995

OVER_STOP = 0.5      # P(stop) after a sentence above this = broken early-stopping
UNDER_STOP = 0.5     # P(stop) after a tool call below this = will not terminate

DEFAULT_WEIGHT = 1.0
RAISED_WEIGHT = 2.0


def read_probe(tag: str) -> dict[str, float | None]:
    if not STOP_CSV.exists():
        return {}
    out: dict[str, float | None] = {}
    for r in csv.DictReader(STOP_CSV.open()):
        if r["label"] != tag:
            continue
        try:
            out[r["probe"]] = float(r["stop_prob"]) if r["stop_prob"] else None
        except ValueError:
            out[r["probe"]] = None
    return out


def read_anomaly(tag: str) -> dict:
    p = REPO / "out" / "exp-058" / "eval" / f"swe_anomalies_{tag}.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except ValueError:
        return {}
    return d[0] if isinstance(d, list) and d else (d if isinstance(d, dict) else {})


def decide(probe: dict, anomaly: dict) -> tuple[float, list[str]]:
    diag = probe.get(DIAG)
    ctrl = probe.get(CTRL)
    mode = anomaly.get("mode")
    why: list[str] = []

    if diag is None and ctrl is None and not mode:
        return DEFAULT_WEIGHT, [
            "no probe and no trajectory for this tag — defaulting to the natural rate "
            f"({DEFAULT_WEIGHT}). This is a fallback, NOT a measurement."]

    over = (diag is not None and diag > OVER_STOP) or mode == "mute"
    under = (ctrl is not None and ctrl < UNDER_STOP) or mode == "loop"

    if diag is not None:
        why.append(f"P(stop | sentence end) = {diag:.5f} (vanilla {VANILLA_DIAG})")
    if ctrl is not None:
        why.append(f"P(stop | after tool call) = {ctrl:.5f} (vanilla {VANILLA_CTRL})")
    if mode:
        why.append(f"agent trajectory mode = {mode}")

    if over and under:
        why.append("CONTRADICTORY: stops too early at a sentence AND fails to stop after "
                   "a tool call. That is not a weight problem — the model has lost the "
                   "position-dependence of the stop decision. Holding at the natural "
                   "rate; raising the weight would make the early-stopping worse.")
        return DEFAULT_WEIGHT, why
    if over:
        why.append(f"stops too early -> the stop signal is already over-weighted; hold at "
                   f"{DEFAULT_WEIGHT} (the natural rate) and look at the corpus, not the "
                   f"weight.")
        return DEFAULT_WEIGHT, why
    if under:
        why.append(f"does not terminate -> raise to {RAISED_WEIGHT}. The curriculum's own "
                   f"corpora are ~1.7x SPARSER in stop signal than the one that produced "
                   f"a 97% loop rate, so this is the expected direction.")
        return RAISED_WEIGHT, why
    why.append(f"termination is in the healthy band -> keep the natural rate "
               f"{DEFAULT_WEIGHT}; reweighting a signal that is already correct only "
               f"risks the other failure mode.")
    return DEFAULT_WEIGHT, why


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="sft32k_sw1")
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()

    probe = read_probe(a.tag)
    anomaly = read_anomaly(a.tag)
    weight, why = decide(probe, anomaly)

    print(f"[stop-weight] tag={a.tag}", file=sys.stderr)
    for line in why:
        print(f"  - {line}", file=sys.stderr)
    print(f"[stop-weight] -> {weight}", file=sys.stderr)

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(
            {"tag": a.tag, "stop_weight": weight, "reasons": why,
             "probe": probe, "anomaly_mode": anomaly.get("mode")}, indent=2) + "\n")

    print(weight)   # stdout: the number, and nothing else
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
