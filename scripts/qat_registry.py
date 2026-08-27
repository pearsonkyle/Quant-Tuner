#!/usr/bin/env python3
"""The QAT run ledger: every training run, its config, and every number measured on it.

    python scripts/qat_registry.py                       # refresh CSV + markdown
    python scripts/qat_registry.py --print               # ... and print the table

Why this exists. A QAT run leaves five artifacts in four places — `train.log` (curves),
`run_config.json` (hyper-parameters), `eval/stop_prob.csv` (termination), a per-tag
tool-call CSV, and swe-mimic's results CSV — keyed three different ways. Comparing two
runs meant joining those by hand, and the join is exactly where a number gets attributed
to the wrong run: `sft32k` and `sft32k_sw1` differ in ONE hyper-parameter, and their rows
sit next to each other in every one of those files under labels that differ by a suffix.

So this is a join, not a new measurement. It reads only; it never re-runs anything. Runs
whose `run_config.json` predates that file (backfilled by hand) are marked in the
`provenance` column so a reconstructed config is never mistaken for a recorded one.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_GLOB = "out/exp-058/trained_*"
GGUF_DIR = REPO / "out" / "exp-057"
STOP_CSV = REPO / "out" / "exp-058" / "eval" / "stop_prob.csv"
TOOLCALL_DIR = REPO / "out" / "exp-058" / "eval"
SWE_CSV = Path("/workspace/swe-mimic/swe_mimic_ternary.csv")

OUT_CSV = REPO / "out" / "exp-058" / "runs.csv"
OUT_MD = REPO / "docs" / "qat_run_history.md"

# The termination probe point that discriminates. `after_tool_call` is the control (a high
# value there is CORRECT); `sentence_period` is where over-stopping shows up.
DIAG_PROBE = "sentence_period"
CTRL_PROBE = "after_tool_call"

STEP_RE = re.compile(
    r"^\[qat\] step (?P<step>\d+)/(?P<total>\d+) loss=(?P<loss>[\d.]+) "
    r"lr=(?P<lr>[\d.eE+-]+)(?: gnorm=(?P<gnorm>[\d.eE+-]+))?"
    r" mem=(?P<mem>[\d.]+)(?:/(?P<peak>[\d.]+))?GiB (?P<sps>[\d.]+)s/step")
VAL_RE = re.compile(r"^\[qat\] step (?P<step>\d+) VAL masked-CE (?P<val>[\d.]+)")
DONE_RE = re.compile(r"^\[qat\] done at step (?P<step>\d+)")
CORPUS_RE = re.compile(
    r"^\[qat\] corpus (?P<n>\d+) windows x (?P<w>\d+) \((?P<masked>\d+)% masked, "
    r"fingerprint (?P<fp>[0-9a-f]+)\); (?P<epochs>[\d.]+) epochs -> (?P<steps>\d+) steps "
    r"@ accum (?P<accum>\d+)")
FLIP_RE = re.compile(r"^\s+(?P<tensor>\S+): flips (?P<pct>[\d.]+)%")


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


RESUMED_RE = re.compile(r"^\[qat\] resumed at step (?P<step>\d+)")


def read_log(path: Path) -> dict:
    """Curves + completion state for ONE log. Tolerates a log still being appended to."""
    out: dict = {"steps": 0, "step_first": None, "total_steps": None,
                 "loss_first": None, "loss_last": None,
                 "loss_peak": None, "val_first": None, "val_last": None, "val_best": None,
                 "s_per_step": None, "mem_peak_gib": None, "complete": False,
                 "flip_pct_max": None, "n_val": 0, "resumed_from": None}
    if not path.exists():
        return out
    flips: dict[str, float] = {}
    # \r: the weight-loading progress bar writes one long line with carriage returns.
    for line in path.read_text(errors="replace").replace("\r", "\n").splitlines():
        if (m := STEP_RE.match(line)):
            loss = _f(m["loss"])
            out["steps"] = int(m["step"])
            out["total_steps"] = int(m["total"])
            if out["loss_first"] is None:
                out["loss_first"] = loss
                out["step_first"] = int(m["step"])
            out["loss_last"] = loss
            out["loss_peak"] = loss if out["loss_peak"] is None else max(out["loss_peak"], loss)
            out["s_per_step"] = _f(m["sps"])
            out["mem_peak_gib"] = _f(m["peak"]) or out["mem_peak_gib"]
        elif (m := VAL_RE.match(line)):
            v = _f(m["val"])
            out["n_val"] += 1
            if out["val_first"] is None:
                out["val_first"] = v
            out["val_last"] = v
            out["val_best"] = v if out["val_best"] is None else min(out["val_best"], v)
        elif (m := DONE_RE.match(line)):
            out["complete"] = True
        elif (m := RESUMED_RE.match(line)):
            out["resumed_from"] = int(m["step"])
        elif (m := CORPUS_RE.match(line)):
            out.update(n_windows=int(m["n"]), window=int(m["w"]),
                       masked_pct=int(m["masked"]), fingerprint=m["fp"],
                       epochs=_f(m["epochs"]), grad_accum=int(m["accum"]))
        elif (m := FLIP_RE.match(line)):
            flips[m["tensor"]] = _f(m["pct"])   # last report wins = latest checkpoint
    if flips:
        out["flip_pct_max"] = max(flips.values())
        out["flip_pct_mean"] = sum(flips.values()) / len(flips)
    return out


def read_legs(run: Path) -> list[dict]:
    """A run is a SEQUENCE of legs, not one log.

    `trained_sft32k` is three: two bf16 attempts that diverged and were renamed
    `train.diverged-run*.log`, then an fp32 leg resumed from step 350. Reading only the
    surviving `train.log` reports its first loss as 0.7676 at step 355 and calls that the
    run's starting loss — off by 350 steps and by a precision change. Every `train*.log`
    in the directory is a leg; the suffix records how it ended.
    """
    legs = []
    for p in sorted(run.glob("train*.log*"), key=lambda q: q.stat().st_mtime):
        log = read_log(p)
        if not log["steps"] and not log["step_first"]:
            continue
        suffix = p.name.replace("train", "", 1).replace(".log", "").strip(".")
        if log["complete"]:
            fate = "complete"
        elif "diverged" in p.name:
            fate = "diverged"
        elif "dead" in p.name:
            fate = "died"
        elif p.name == "train.log":
            fate = "running"
        else:
            fate = suffix or "killed"
        legs.append({"file": p.name, "fate": fate, **log})
    return legs


def read_config(run: Path) -> tuple[dict, str]:
    """(config, provenance). Prefers the trainer's own record; falls back to the log."""
    p = run / "run_config.json"
    if p.exists():
        try:
            return json.loads(p.read_text()), "recorded"
        except ValueError:
            pass
    p = run / "run_config.backfill.json"
    if p.exists():
        try:
            return json.loads(p.read_text()), "backfilled"
        except ValueError:
            pass
    return {}, "log-only"


def read_stop_probs() -> dict[str, dict[str, float | None]]:
    if not STOP_CSV.exists():
        return {}
    out: dict[str, dict[str, float | None]] = {}
    with STOP_CSV.open() as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["label"], {})[r["probe"]] = _f(r["stop_prob"])
    return out


def read_toolcall(tag: str) -> dict:
    p = TOOLCALL_DIR / f"toolcall_{tag}.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(p.open()))
    if not rows:
        return {}
    r = rows[-1]
    return {"tool_sel_acc": _f(r.get("tool_selection_acc")),
            "param_acc": _f(r.get("param_acc_mean")),
            "schema_valid": _f(r.get("schema_valid_rate"))}


def read_swe(tag: str) -> dict:
    """Join swe-mimic by label. Labels there are uppercase with a quant suffix."""
    if not SWE_CSV.exists():
        return {}
    want = tag.upper()
    hit = None
    for r in csv.DictReader(SWE_CSV.open()):
        lab = r.get("label", "")
        # exact stem match only: SFT32K must not claim SFT32K_SW1's row.
        if lab.upper().split("-")[0] == want:
            hit = r      # last wins — a re-run supersedes
    if not hit:
        return {}
    return {"swe_resolved": _f(hit.get("resolved")),
            "swe_patch": _f(hit.get("patch_produced")),
            "swe_steps": _f(hit.get("steps")),
            "swe_out_tokens": _f(hit.get("out_tokens")),
            "swe_exit": hit.get("exit_status"),
            "swe_tool_err_rate": _f(hit.get("tool_err_rate"))}


COLUMNS = [
    "tag", "provenance", "status", "steps", "total_steps",
    "legs", "legs_diverged", "resumed_from", "step_first",
    "corpus", "window", "n_windows", "masked_pct", "fingerprint",
    "lr", "stop_weight", "epochs", "grad_accum", "optim", "train_layers",
    "dtype", "compute_dtype", "matmul_precision", "resume", "device",
    "loss_first", "loss_last", "loss_peak", "val_first", "val_best", "val_last",
    "flip_pct_max", "flip_pct_mean", "s_per_step", "mem_peak_gib",
    "stop_p_sentence", "stop_p_after_tool",
    "tool_sel_acc", "param_acc", "schema_valid",
    "swe_resolved", "swe_patch", "swe_steps", "swe_out_tokens", "swe_exit",
    "gguf", "git_commit", "started_utc",
]


def collect(all_legs: dict[str, list[dict]] | None = None) -> list[dict]:
    stops = read_stop_probs()
    all_legs = {} if all_legs is None else all_legs
    rows = []
    for run in sorted(REPO.glob(RUN_GLOB)):
        if not run.is_dir():
            continue
        tag = run.name.removeprefix("trained_")
        cfg, prov = read_config(run)
        legs = read_legs(run)
        log = read_log(run / "train.log")
        gguf = GGUF_DIR / f"Ternary-Bonsai-8B-{tag}-Q2_0.gguf"
        all_legs[tag] = legs

        status = "complete" if log["complete"] else (
            "running" if log["steps"] else "empty")
        if not log["complete"] and log["steps"] and log["total_steps"]:
            status = f"running {log['steps']}/{log['total_steps']}"

        row = {
            "tag": tag, "provenance": prov, "status": status,
            "steps": log["steps"], "total_steps": log["total_steps"],
            "corpus": Path(cfg.get("corpus", "")).name or None,
            "window": cfg.get("window") or log.get("window"),
            "n_windows": cfg.get("n_windows") or log.get("n_windows"),
            "masked_pct": log.get("masked_pct"),
            "fingerprint": cfg.get("fingerprint") or log.get("fingerprint"),
            "lr": cfg.get("lr"), "stop_weight": cfg.get("stop_weight"),
            "epochs": cfg.get("epochs") or log.get("epochs"),
            "grad_accum": cfg.get("grad_accum") or log.get("grad_accum"),
            "optim": cfg.get("optim"), "train_layers": cfg.get("train_layers"),
            "dtype": cfg.get("dtype"), "compute_dtype": cfg.get("compute_dtype"),
            "matmul_precision": cfg.get("matmul_precision"),
            "resume": Path(cfg["resume"]).name if cfg.get("resume") else None,
            "device": cfg.get("device_resolved") or cfg.get("device"),
            # From the FIRST leg — the run's actual starting loss. Taking it from the
            # surviving log reports a resumed leg's mid-run value as the start.
            "loss_first": (legs[0]["loss_first"] if legs else log["loss_first"]),
            "step_first": (legs[0]["step_first"] if legs else log["step_first"]),
            "legs": len(legs) or None,
            "legs_diverged": sum(1 for x in legs if x["fate"] == "diverged") or None,
            "resumed_from": log["resumed_from"],
            "loss_last": log["loss_last"],
            "loss_peak": log["loss_peak"], "val_first": log["val_first"],
            "val_best": log["val_best"], "val_last": log["val_last"],
            "flip_pct_max": log["flip_pct_max"],
            "flip_pct_mean": log.get("flip_pct_mean"),
            "s_per_step": log["s_per_step"], "mem_peak_gib": log["mem_peak_gib"],
            "stop_p_sentence": stops.get(tag, {}).get(DIAG_PROBE),
            "stop_p_after_tool": stops.get(tag, {}).get(CTRL_PROBE),
            "gguf": gguf.name if gguf.exists() else None,
            "git_commit": (cfg.get("git_commit") or "")[:9] or None,
            "started_utc": cfg.get("started_utc"),
        }
        row.update(read_toolcall(tag))
        row.update(read_swe(tag))
        rows.append(row)

    # The shipped weights are the control every run is read against, and they have no run
    # directory — but they DO have probe and swe-mimic rows. Omitting them would leave the
    # ledger unable to answer "did training make this worse than doing nothing?".
    if "vanilla" in stops or (SWE_CSV.exists() and read_swe("vanilla")):
        base = {c: None for c in COLUMNS}
        base.update(tag="vanilla (shipped)", provenance="control", status="untrained",
                    stop_p_sentence=stops.get("vanilla", {}).get(DIAG_PROBE),
                    stop_p_after_tool=stops.get("vanilla", {}).get(CTRL_PROBE),
                    gguf="Ternary-Bonsai-8B-vanilla-Q2_0.gguf")
        base.update(read_toolcall("vanilla"))
        base.update(read_swe("vanilla"))
        rows.insert(0, base)
    return rows


def fmt(v, nd=4) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        if v != v:
            return "—"
        if v and (abs(v) < 1e-3):
            return f"{v:.1e}"
        return f"{v:.{nd}g}"
    return str(v)


MD_GROUPS = [
    ("Run", ["tag", "status", "provenance", "legs", "legs_diverged", "resumed_from"]),
    ("Config", ["corpus", "window", "lr", "stop_weight", "epochs", "optim",
                "matmul_precision", "resume"]),
    ("Training", ["loss_first", "loss_last", "loss_peak", "val_best", "val_last",
                  "flip_pct_max", "s_per_step", "mem_peak_gib"]),
    ("Termination", ["stop_p_sentence", "stop_p_after_tool"]),
    ("Tool-call", ["tool_sel_acc", "param_acc", "schema_valid"]),
    ("SWE-mimic", ["swe_resolved", "swe_patch", "swe_steps", "swe_out_tokens", "swe_exit"]),
]


def legs_markdown(all_legs: dict[str, list[dict]]) -> str:
    """Every attempt, including the ones that failed.

    A run that diverged twice and succeeded on the third try is not the same evidence as
    one that succeeded first time, and the difference is invisible once the failed logs
    are renamed out of the way.
    """
    multi = {t: L for t, L in all_legs.items() if len(L) > 1}
    if not multi:
        return ""
    out = ["## Training legs", "",
           "Runs that took more than one attempt. `fate` comes from how the log was left:",
           "a `diverged`/`dead` leg was renamed aside, so it survives here rather than",
           "being overwritten by the attempt that worked.", "",
           "| tag | leg | fate | steps | loss first→last | peak loss | resumed from |",
           "|---|---|---|---|---|---|---|"]
    for tag, legs in multi.items():
        for lg in legs:
            span = f"{fmt(lg['loss_first'])} → {fmt(lg['loss_last'])}"
            out.append(f"| {tag} | `{lg['file']}` | {lg['fate']} | "
                       f"{fmt(lg['step_first'])}–{fmt(lg['steps'])} | {span} | "
                       f"{fmt(lg['loss_peak'])} | {fmt(lg['resumed_from'])} |")
    return "\n".join(out) + "\n\n"


def to_markdown(rows: list[dict], all_legs: dict[str, list[dict]] | None = None) -> str:
    lines = [
        "# QAT run history",
        "",
        "Generated by `scripts/qat_registry.py` — do not hand-edit; it is a join over",
        "`run_config.json`, `train.log`, `eval/stop_prob.csv`, the per-tag tool-call CSV",
        "and swe-mimic's results. Re-run it after any training or eval finishes.",
        "",
        "`provenance` says where the config came from: **recorded** (the trainer wrote it),",
        "**backfilled** (reconstructed by hand from the launch script — trust it less), or",
        "**log-only** (nothing but `train.log` was available).",
        "",
        "`stop_p_sentence` is P(`<|im_end|>`) after a completed sentence — vanilla ~0.009 is",
        "healthy, ~0.97 means the model stops mid-task. `stop_p_after_tool` is the control,",
        "where a HIGH value is correct.",
        "",
    ]
    for title, cols in MD_GROUPS:
        cols = [c for c in cols if any(r.get(c) not in (None, "") for r in rows)]
        if not cols:
            continue
        lines += [f"## {title}", "",
                  "| " + " | ".join(["tag" if title == "Run" else "tag"] +
                                    [c for c in cols if c != "tag"]) + " |",
                  "|" + "---|" * (len(cols) if "tag" in cols else len(cols) + 1)]
        for r in rows:
            cells = [str(r["tag"])] + [fmt(r.get(c)) for c in cols if c != "tag"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n" + legs_markdown(all_legs or {})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=OUT_CSV)
    ap.add_argument("--md", type=Path, default=OUT_MD)
    ap.add_argument("--print", dest="do_print", action="store_true")
    a = ap.parse_args()

    all_legs: dict[str, list[dict]] = {}
    rows = collect(all_legs)
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with a.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    a.md.parent.mkdir(parents=True, exist_ok=True)
    a.md.write_text(to_markdown(rows, all_legs))
    print(f"[registry] {len(rows)} runs -> {a.csv} and {a.md}")
    if a.do_print:
        print()
        print(to_markdown(rows, all_legs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
