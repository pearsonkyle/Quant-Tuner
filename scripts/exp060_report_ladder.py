"""Render the exp-060 ladder results, and optionally A/B two calibration runs.

Reads the per-eval results CSVs that ``exp060_quants_qwen38.py`` appends to
(``results.csv``, ``results.general.csv``, …) and renders a markdown table per eval:
bpw, size, PPL, KLD vs the F16 baseline, and top-token agreement.

With ``--compare <other-run>`` it renders the delta between two runs that share the same
F16 and the same eval corpora/baselines — so the only thing that differs is the calibration
corpus, and the delta is attributable to it.

    PYTHONPATH=src .venv/bin/python scripts/exp060_report_ladder.py --run exp-060-32k
    PYTHONPATH=src .venv/bin/python scripts/exp060_report_ladder.py \\
        --run exp-060-32k --compare exp-060-32k-r16 \\
        --label-a "reasoning 15%" --label-b "reasoning 22%"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# eval name -> results csv (mirrors EVALS in exp060_quants_qwen38.py)
EVAL_CSVS = {
    "external": "results.csv",
    "general": "results.general.csv",
    "tools": "results.tools.csv",
    "agentic": "results.agentic.csv",
    "broad": "results.broad.csv",
    "redteam": "results.redteam.csv",
    "cal8k": "results.cal8k.csv",
}
ROW_ORDER = ["IQ2_M", "IQ3_M", "IQ4_XS", "Q5_K_M", "Q2_K"]

# Evals whose absolute PPL is off-distribution: llama-perplexity has no --parse-special,
# so chat markers tokenize as plain BPE. Quant-vs-quant on the same file stays valid.
CHAT_TEMPLATED = {"tools", "agentic", "broad", "redteam", "cal8k"}


def _quant_of(row: dict) -> str:
    model = row.get("model", "")
    for q in ROW_ORDER:
        if q in model:
            return q
    return model.split("|")[1] if "|" in model else model


def read_run(root: Path) -> dict[str, dict[str, dict]]:
    """{eval_name: {quant: row}}"""
    out: dict[str, dict[str, dict]] = {}
    for name, fname in EVAL_CSVS.items():
        p = root / fname
        if not p.exists():
            continue
        rows: dict[str, dict] = {}
        with p.open() as f:
            for r in csv.DictReader(f):
                rows[_quant_of(r)] = r
        if rows:
            out[name] = rows
    return out


def _f(row: dict, key: str) -> float | None:
    v = (row or {}).get(key)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _fmt(v: float | None, spec: str) -> str:
    return "—" if v is None else format(v, spec)


def render_single(run: str, data: dict[str, dict[str, dict]]) -> str:
    lines = [f"# exp-060 ladder — `{run}`", ""]
    for name in EVAL_CSVS:
        if name not in data:
            continue
        rows = data[name]
        lines.append(f"## eval: {name}"
                     + ("  _(chat-templated: quant-vs-quant only, absolute PPL is "
                        "off-distribution)_" if name in CHAT_TEMPLATED else ""))
        lines.append("")
        lines.append("| rung | size GiB | bpw | PPL | PPL ratio | median KLD | mean KLD | "
                     "top-token agree % |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for q in ROW_ORDER:
            if q not in rows:
                continue
            r = rows[q]
            lines.append(
                f"| {q} | {_fmt(_f(r,'size_gib'), '.2f')} | {_fmt(_f(r,'bpw'), '.3f')} | "
                f"{_fmt(_f(r,'ppl'), '.4f')} | {_fmt(_f(r,'ppl_ratio'), '.4f')} | "
                f"{_fmt(_f(r,'median_kld'), '.5f')} | {_fmt(_f(r,'mean_kld'), '.5f')} | "
                f"{_fmt(_f(r,'same_top_p'), '.2f')} |")
        lines.append("")
    return "\n".join(lines)


def render_compare(run_a: str, run_b: str, a: dict, b: dict,
                   label_a: str, label_b: str) -> str:
    lines = [f"# exp-060 A/B — calibration corpus effect", "",
             f"* **A** = `{run_a}` ({label_a})", f"* **B** = `{run_b}` ({label_b})", "",
             "Same F16, same eval corpora, same F16 baselines — the only difference is the "
             "calibration corpus, so the delta is attributable to it. "
             "Lower KLD is better; higher top-token agreement is better.", ""]
    for name in EVAL_CSVS:
        if name not in a and name not in b:
            continue
        lines.append(f"## eval: {name}"
                     + ("  _(chat-templated: quant-vs-quant only)_"
                        if name in CHAT_TEMPLATED else ""))
        lines.append("")
        lines.append("| rung | medKLD A | medKLD B | Δ KLD | top-tok A | top-tok B | "
                     "Δ top-tok | PPL A | PPL B |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for q in ROW_ORDER:
            ra, rb = a.get(name, {}).get(q), b.get(name, {}).get(q)
            if not ra and not rb:
                continue
            ka, kb = _f(ra, "median_kld"), _f(rb, "median_kld")
            ta, tb = _f(ra, "same_top_p"), _f(rb, "same_top_p")
            dk = None if (ka is None or kb is None) else kb - ka
            dt = None if (ta is None or tb is None) else tb - ta
            mark = ""
            if dk is not None:
                mark = " ✅" if dk < 0 else (" ❌" if dk > 0 else "")
            lines.append(
                f"| {q} | {_fmt(ka,'.5f')} | {_fmt(kb,'.5f')} | "
                f"{_fmt(dk,'+.5f')}{mark} | {_fmt(ta,'.2f')} | {_fmt(tb,'.2f')} | "
                f"{_fmt(dt,'+.2f')} | {_fmt(_f(ra,'ppl'),'.4f')} | "
                f"{_fmt(_f(rb,'ppl'),'.4f')} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="exp-060-32k")
    p.add_argument("--compare", default=None, help="second run to A/B against")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    a = read_run(REPO / "out" / args.run)
    if not a:
        print(f"no results CSVs yet under out/{args.run}")
        return 1
    if args.compare:
        b = read_run(REPO / "out" / args.compare)
        text = render_compare(args.run, args.compare, a, b, args.label_a, args.label_b)
    else:
        text = render_single(args.run, a)
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
