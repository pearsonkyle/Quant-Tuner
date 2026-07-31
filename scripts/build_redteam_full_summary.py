#!/usr/bin/env python3
"""Render the FULL red-team sweep into a combined markdown table.

The full sweep runs single-rep with a large apvt, so the meaningful uncertainty
is a binomial (Wilson 95%) confidence interval on each pass_rate computed from
the pass/fail COUNTS — not a rep-to-rep stdev (there's only one rep). Wilson is
used because it's well-behaved near 0/1 and for small n.

Sources (whatever exists is used; missing models are skipped so this can run
mid-sweep):
  * overall counts        <- gemma_full_results.csv  (per-rep rows)
  * per-category counts   <- run_full.log            (render_summary CATEGORY tables)
  * per-vulnerability rate <- out/redteam/full/redteam_*_agg_*.json

Usage:
  .venv/bin/python scripts/build_redteam_full_summary.py \
      --csv out/redteam/gemma_full_results.csv \
      --log out/redteam/run_full.log \
      --agg-dir out/redteam/full \
      --out out/redteam/gemma_redteam_full_summary.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path

ORDER = ["IQ2_M", "IQ3_M", "IQ4_XS", "Q5_K_S"]
BPW = {"IQ2_M": 2.85, "IQ3_M": 3.76, "IQ4_XS": 4.36, "Q5_K_S": 5.55}
CATEGORIES = ["Responsible AI", "Data Privacy", "Safety"]

# deepteam prints one "Mitigation Rate" line per attack *method* (no parenthetical
# vulnerability type). These are the display names to pull out of run_full.log.
ATTACK_METHODS = [
    "Leetspeak", "ROT-13", "Gray Box", "Roleplay", "Math Problem", "Base64",
    "Multilingual", "Prompt Injection", "Emotional Manipulation",
    "Authority Escalation", "Linear Jailbreaking", "Crescendo Jailbreaking",
    "Tree Jailbreaking", "Baseline",
]
_ATTACK_RE = re.compile(
    r"\|\s*([A-Za-z0-9 \-]+?)\s*\|\s*Mitigation Rate:\s*[\d.]+%\s*\((\d+)/(\d+)\)"
)
_TGT_RE = re.compile(r"Red-teaming target:\s*\S*?(IQ2_M|IQ3_M|IQ4_XS|Q5_K_S)")


def wilson(passing: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (point, lo, hi) Wilson score interval for passing/total."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = passing / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def key_of(model: str) -> str | None:
    for k in ORDER:
        if k in model:
            return k
    return None


def parse_overall(csv_path: Path) -> dict[str, tuple[int, int]]:
    """model-key -> (passing, total) from the per-rep CSV."""
    out: dict[str, tuple[int, int]] = {}
    if not csv_path.exists():
        return out
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            k = key_of(row["model"])
            if k:
                out[k] = (int(row["n_passing"]),
                          int(row["n_passing"]) + int(row["n_failing"]))
    return out


_CAT_RE = re.compile(
    r"^\s*(Responsible AI|Data Privacy|Safety)\s+\d+%\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
)


def parse_categories(log_path: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """model-key -> {category -> (passing, total)} from render_summary blocks."""
    out: dict[str, dict[str, tuple[int, int]]] = {}
    if not log_path.exists():
        return out
    cur: str | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.search(r"Red-teaming target:\s*\S*?(IQ2_M|IQ3_M|IQ4_XS|Q5_K_S)", line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, {})
            continue
        if cur:
            cm = _CAT_RE.match(line)
            if cm:
                cat, passing, fail, _err, total = cm.groups()
                out[cur][cat] = (int(passing), int(total))
    return out


def parse_vuln_rates(agg_dir: Path) -> dict[str, dict[str, dict]]:
    """model-key -> {vuln -> {category, pass_rate}} from aggregate JSONs."""
    out: dict[str, dict[str, dict]] = {}
    for f in glob.glob(str(agg_dir / "redteam_*_agg_*.json")):
        with open(f) as fh:
            d = json.load(fh)
        k = key_of(d["model"])
        if k:
            out[k] = d["by_vulnerability"]
    return out


def parse_attacks(log_path: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """model-key -> {attack_method -> (passing, total)} from render_summary blocks.

    An attack-method row is a Mitigation-Rate line whose label has no
    parenthetical vulnerability type (those are the per-vuln-type rows).
    """
    out: dict[str, dict[str, tuple[int, int]]] = {}
    if not log_path.exists():
        return out
    cur: str | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = _TGT_RE.search(line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, {})
            continue
        if not cur:
            continue
        r = _ATTACK_RE.search(line)
        if r and r.group(1).strip() in ATTACK_METHODS:
            out[cur][r.group(1).strip()] = (int(r.group(2)), int(r.group(3)))
    return out


def ci_cell(passing: int | None, total: int | None) -> str:
    if passing is None or total is None or total == 0:
        return "—"
    p, lo, hi = wilson(passing, total)
    return f"{p*100:.1f}% [{lo*100:.0f}–{hi*100:.0f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("out/redteam/gemma_full_results.csv"))
    ap.add_argument("--log", type=Path, default=Path("out/redteam/run_full.log"))
    ap.add_argument("--agg-dir", type=Path, default=Path("out/redteam/full"))
    ap.add_argument("--out", type=Path, default=Path("out/redteam/gemma_redteam_full_summary.md"))
    args = ap.parse_args()

    overall = parse_overall(args.csv)
    cats = parse_categories(args.log)
    vulns = parse_vuln_rates(args.agg_dir)
    attacks = parse_attacks(args.log)
    done = [k for k in ORDER if k in overall]
    if not done:
        print("No completed models yet.")
        return 1

    def hdr() -> list[str]:
        return ["| Metric | " + " | ".join(f"**{k}**" for k in done) + " |",
                "|:---|" + "".join(":---:|" for _ in done)]

    L: list[str] = []
    L.append("# Gemma-4-31B quant red-team — FULL sweep")
    L.append("")
    L.append(f"Single-rep, large-coverage sweep ({len(done)}/4 models complete). "
             "Each cell is **pass_rate% [Wilson 95% CI]** from the pass/fail counts "
             "(higher = safer). With one rep and hundreds of cases, the Wilson CI — not a "
             "rep stdev — is the right uncertainty; it tightens as the case count grows.")
    L.append("")
    L.append("| | " + " | ".join(f"**{k}**" for k in done) + " |")
    L.append("|:---|" + "".join(":---:|" for _ in done))
    L.append("| BPW | " + " | ".join(f"{BPW[k]:.2f}" for k in done) + " |")
    L.append("| Cases | " + " | ".join(str(overall[k][1]) for k in done) + " |")
    L.append("| **Overall** | " + " | ".join(ci_cell(*overall[k]) for k in done) + " |")
    for cat in CATEGORIES:
        cells = " | ".join(ci_cell(*cats.get(k, {}).get(cat, (None, None))) for k in done)
        L.append(f"| **{cat}** | {cells} |")
    L.append("")

    # ── Per-vulnerability rates (no counts persisted -> rate only) ───────────
    all_vulns: list[str] = []
    for k in done:
        for v in vulns.get(k, {}):
            if v not in all_vulns:
                all_vulns.append(v)
    if all_vulns:
        L.append("### Per-vulnerability pass rate (point estimate)")
        L.append("")
        L.append("| Vulnerability | " + " | ".join(f"**{k}**" for k in done) + " |")
        L.append("|:---|" + "".join(":---:|" for _ in done))
        for v in all_vulns:
            cells = []
            for k in done:
                d = vulns.get(k, {}).get(v)
                cells.append(f"{d['pass_rate_mean']*100:.1f}%" if d else "—")
            L.append(f"| {v} | " + " | ".join(cells) + " |")
        L.append("")

    # ── Per-attack-method pass rate (which jailbreak vectors break through) ───
    def attack_rate(a: str) -> float:
        tot = sum(attacks.get(k, {}).get(a, (0, 0))[1] for k in done)
        pas = sum(attacks.get(k, {}).get(a, (0, 0))[0] for k in done)
        return pas / tot if tot else 1.0

    present = [a for a in ATTACK_METHODS if any(a in attacks.get(k, {}) for k in done)]
    present.sort(key=attack_rate)  # weakest defense first
    if present:
        L.append("### Per-attack-method pass rate (which vectors break through)")
        L.append("")
        L.append("Sorted weakest → strongest defense. `n/N` = passed / total for that "
                 "vector; higher % = the quant resisted more of that attack style.")
        L.append("")
        L.append("| Attack vector | " + " | ".join(f"**{k}**" for k in done) + " | mean |")
        L.append("|:---|" + "".join(":---:|" for _ in done) + ":---:|")
        for a in present:
            cells = []
            for k in done:
                pt = attacks.get(k, {}).get(a)
                cells.append(f"{pt[0]/pt[1]*100:.0f}% ({pt[0]}/{pt[1]})" if pt else "—")
            L.append(f"| {a} | " + " | ".join(cells) + f" | {attack_rate(a)*100:.0f}% |")
        L.append("")

    args.out.write_text("\n".join(L) + "\n")
    print(f"WROTE {args.out}  ({len(done)}/4 models)\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
