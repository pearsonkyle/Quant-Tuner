"""Experiment 007: Q4_K_M vs IQ4_NL calibration sweep on google/gemma-4-E4B-it.

Reuses the F16 GGUF, KLD baseline, and the 5 imatrices already produced under
exp-001 (`custom`, `wiki`, `mixed512`, `mixed2k`, `mixed8k`). Builds 5 IQ4_NL
GGUFs against those same imatrices and benches each. Q4_K_M rows are sourced
from `out/exp-001/google__gemma-4-E4B-it/results.csv`; the F16 row is
constructed from the baseline log (KLD=0, same_top_p=100% by definition).

Output: `out/exp-007/results.csv` (one row per (quant, cell)) plus a rendered
markdown table in the experiment README.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import llama_cpp  # noqa: F401  (kept for symmetry)
from quant_tuner.quantize import gguf

SRC = REPO / "out" / "exp-001" / "google__gemma-4-E4B-it"
WORK = REPO / "out" / "exp-007"
LOGS = WORK / "logs"
F16 = SRC / "model-f16.gguf"
BASE_KLD = SRC / "baseline.kld"
EVAL_DS = SRC / "corpus.eval.txt"
EXP001_CSV = SRC / "results.csv"
EVAL_CTX = 4096

CELLS = [
    # (cell, imatrix_filename, dataset_label_used_in_exp-001)
    ("custom", "imatrix-custom.gguf", "custom"),
    ("wiki", "imatrix-wiki.gguf", "wiki.test.raw"),
    ("mixed512", "imatrix-mixed512.gguf", "500k-custom+wiki (ctx=512)"),
    ("mixed2k", "imatrix-mixed2k.gguf", "500k-custom+wiki (ctx=2048)"),
    ("mixed8k", "imatrix-mixed8k.gguf", "500k-custom+wiki (ctx=8192)"),
]


def _f16_ppl_from_baseline_log() -> float | None:
    log_file = SRC / "logs" / "baseline.log"
    if not log_file.exists():
        return None
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", log_file.read_text())
    return float(m.group(1)) if m else None


def _q4km_rows() -> dict[str, dict]:
    """Index exp-001 rows for gemma by dataset label → row dict."""
    out: dict[str, dict] = {}
    with EXP001_CSV.open() as f:
        for row in csv.DictReader(f):
            label = row["model"]
            if not label.startswith("google/gemma-4-E4B-it|"):
                continue
            parts = label.split("|", 2)
            if len(parts) != 3:
                continue
            _, _, dataset = parts
            out[dataset] = row
    return out


def main() -> int:
    for required in (F16, BASE_KLD, EVAL_DS, EXP001_CSV):
        if not required.exists():
            print(f"missing prerequisite: {required}", file=sys.stderr)
            return 1
    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    out_csv = WORK / "results.csv"

    n_params = bpw_mod.n_params(F16)
    log(f"f16 n_params = {n_params:,.0f}")

    # FP16 self-row
    fp16_ppl = _f16_ppl_from_baseline_log()
    fp16_size_gib = F16.stat().st_size / (1024 ** 3)
    fp16_bpw = (F16.stat().st_size * 8) / n_params
    log(f"f16 size = {fp16_size_gib:.3f} GiB  bpw = {fp16_bpw:.3f}  ppl = {fp16_ppl}")

    # Q4_K_M rows (reuse exp-001)
    q4km = _q4km_rows()

    # Build + bench IQ4_NL for each cell
    iq4_rows: list[dict] = []
    for cell, imat_name, _ds in CELLS:
        imat = SRC / imat_name
        qpath = WORK / f"IQ4_NL-{cell}.gguf"
        if not imat.exists():
            print(f"missing imatrix: {imat}", file=sys.stderr)
            return 1
        with phase(f"IQ4_NL {cell}"):
            step(f"quantize IQ4_NL ({cell})", qpath,
                 lambda q=qpath, s=imat, n=cell: gguf.quantize(
                     F16, q, "IQ4_NL", imatrix=s,
                     log=LOGS / f"quantize-{n}.log"))
            label = f"google/gemma-4-E4B-it|IQ4_NL|imatrix|{cell}"
            with phase(f"bench {label}"):
                row = runner.bench_one(
                    qpath, label,
                    reference_n_params=n_params,
                    eval_dataset=EVAL_DS,
                    eval_baseline=BASE_KLD,
                    eval_ctx=EVAL_CTX,
                    log_dir=LOGS,
                    suite="kld",
                )
                runner.append_row(out_csv, row)
                iq4_rows.append({
                    "cell": cell,
                    "size_gib": row.size_gib, "bpw": row.bpw,
                    "ppl": row.ppl, "mean_kld": row.mean_kld,
                    "same_top_p": row.same_top_p,
                })
                log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                    f"ppl={row.ppl} mean_kld={row.mean_kld} "
                    f"same_top_p={row.same_top_p}")

    # Render markdown table
    md_path = WORK / "table.md"
    lines = [
        "| quant | technique | dataset | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |",
        "|---|---|---|---|---|---|---|---|",
        f"| FP16   | none    | —             | {fp16_size_gib:.2f} | {fp16_bpw:.3f} | "
        f"{fp16_ppl:.4f} | 0.00000 | 100.0000 |",
    ]
    cell_to_ds_label = {
        "wiki": "wiki.test.raw",
        "custom": "custom",
        "mixed512": "500k-custom+wiki (ctx=512)",
        "mixed2k": "500k-custom+wiki (ctx=2048)",
        "mixed8k": "500k-custom+wiki (ctx=8192)",
    }
    table_order = ["wiki", "custom", "mixed512", "mixed2k", "mixed8k"]

    def _fmt(v: str | float | None, places: int) -> str:
        try:
            return f"{float(v):.{places}f}"
        except (TypeError, ValueError):
            return "—"

    for cell in table_order:
        ds = cell_to_ds_label[cell]
        r = q4km.get(ds)
        if r is None:
            lines.append(f"| Q4_K_M | imatrix | {ds} | — | — | — | — | — |")
        else:
            lines.append(
                f"| Q4_K_M | imatrix | {ds} | "
                f"{_fmt(r['size_gib'], 2)} | {_fmt(r['bpw'], 3)} | "
                f"{_fmt(r['ppl'], 4)} | {_fmt(r['mean_kld'], 5)} | "
                f"{_fmt(r['same_top_p'], 4)} |"
            )
    by_cell = {r["cell"]: r for r in iq4_rows}
    for cell in table_order:
        r = by_cell.get(cell)
        ds = cell_to_ds_label[cell]
        if r is None:
            lines.append(f"| IQ4_NL | imatrix | {ds} | — | — | — | — | — |")
        else:
            lines.append(
                f"| IQ4_NL | imatrix | {ds} | "
                f"{_fmt(r['size_gib'], 2)} | {_fmt(r['bpw'], 3)} | "
                f"{_fmt(r['ppl'], 4)} | {_fmt(r['mean_kld'], 5)} | "
                f"{_fmt(r['same_top_p'], 4)} |"
            )

    md_path.write_text("\n".join(lines) + "\n")
    log(f"table -> {md_path.relative_to(REPO)}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
