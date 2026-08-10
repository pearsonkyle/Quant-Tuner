"""Experiment 008: Q4_K_M vs IQ4_NL calibration sweep on Jackrong/Qwopus3.5-9B-Coder.

Mirror of exp-007 (gemma-4-E4B-it) but on the Qwen3.5-class 9B model.
Exp-001 produced the F16 GGUF, baseline KLD (ctx=8192), and 3 imatrices
(custom, wiki, mixed8k). This script:

  1. Builds the two missing imatrices for the mixed corpus at
     ctx=2048 and ctx=512 (alongside the existing ones).
  2. Builds the two missing Q4_K_M cells (mixed2k, mixed512) under
     out/exp-001/... and benches them so all 5 Q4_K_M cells are
     comparable to the existing custom/wiki/mixed8k rows.
  3. Builds 5 IQ4_NL GGUFs (one per imatrix) under out/exp-008/ and
     benches each against the existing baseline.
  4. Renders the 11-row table to out/exp-008/table.md.
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
from quant_tuner.models import llama_cpp
from quant_tuner.quantize import gguf

SRC = REPO / "out" / "exp-001" / "Jackrong__Qwopus3.5-9B-Coder"
WORK = REPO / "out" / "exp-008"
LOGS = WORK / "logs"
F16 = SRC / "model-f16.gguf"
BASE_KLD = SRC / "baseline.kld"
EVAL_DS = SRC / "corpus.eval.txt"
MIXED = SRC / "corpus.mixed8k.txt"
EXP001_CSV = SRC / "results.csv"
AGG_CSV = REPO / "out" / "exp-001" / "results.csv"
EVAL_CTX = 8192  # Qwen-class vocab handles this fine

# (cell, imatrix_filename, ctx_for_imatrix_build, dataset_label)
CELLS = [
    ("custom",   "imatrix-custom.gguf",    512, "custom"),
    ("wiki",     "imatrix-wiki.gguf",      512, "wiki.test.raw"),
    ("mixed512", "imatrix-mixed512.gguf",  512, "500k-custom+wiki (ctx=512)"),
    ("mixed2k",  "imatrix-mixed2k.gguf",  2048, "500k-custom+wiki (ctx=2048)"),
    ("mixed8k",  "imatrix-mixed8k.gguf",  8192, "500k-custom+wiki (ctx=8192)"),
]


def _f16_ppl_from_baseline_log() -> float | None:
    log_file = SRC / "logs" / "baseline.log"
    if not log_file.exists():
        return None
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", log_file.read_text())
    return float(m.group(1)) if m else None


def _existing_rows(label_prefix: str) -> dict[str, dict]:
    """Index a results.csv by dataset label → row dict."""
    out: dict[str, dict] = {}
    if not EXP001_CSV.exists():
        return out
    with EXP001_CSV.open() as f:
        for row in csv.DictReader(f):
            label = row["model"]
            if not label.startswith(label_prefix):
                continue
            parts = label.split("|")
            if not parts:
                continue
            dataset = parts[-1]
            out[dataset] = row
    return out


def main() -> int:
    for required in (F16, BASE_KLD, EVAL_DS, MIXED):
        if not required.exists():
            print(f"missing prerequisite: {required}", file=sys.stderr)
            return 1
    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    out_csv = WORK / "results.csv"

    n_params = bpw_mod.n_params(F16)
    log(f"f16 n_params = {n_params:,.0f}")

    fp16_ppl = _f16_ppl_from_baseline_log()
    fp16_size_gib = F16.stat().st_size / (1024 ** 3)
    fp16_bpw = (F16.stat().st_size * 8) / n_params
    log(f"f16 size = {fp16_size_gib:.3f} GiB  bpw = {fp16_bpw:.3f}  ppl = {fp16_ppl}")

    # Stage 1 + 2: ensure all 5 imatrices and Q4_K_M cells exist + benched
    q4km_existing = _existing_rows("Jackrong/Qwopus3.5-9B-Coder|imatrix|")
    for cell, imat_name, imat_ctx, ds_label in CELLS:
        imat = SRC / imat_name
        if not imat.exists():
            with phase(f"build imatrix {cell} (ctx={imat_ctx})"):
                step(f"llama-imatrix {cell}", imat,
                     lambda i=imat, c=imat_ctx, n=cell: llama_cpp.imatrix(
                         F16, MIXED, i, ctx=c, log=LOGS / f"imatrix-{n}.log"))

        q4km_path = SRC / f"Q4_K_M-{cell}.gguf"
        if ds_label not in q4km_existing:
            with phase(f"Q4_K_M {cell}"):
                step(f"quantize Q4_K_M ({cell})", q4km_path,
                     lambda q=q4km_path, s=imat, n=cell: gguf.quantize(
                         F16, q, "Q4_K_M", imatrix=s,
                         log=LOGS / f"quantize-q4km-{n}.log"))
                label = f"Jackrong/Qwopus3.5-9B-Coder|imatrix|{ds_label}"
                with phase(f"bench {label}"):
                    row = runner.bench_one(
                        q4km_path, label,
                        reference_n_params=n_params,
                        eval_dataset=EVAL_DS,
                        eval_baseline=BASE_KLD,
                        eval_ctx=EVAL_CTX,
                        log_dir=LOGS,
                        suite="kld",
                    )
                    runner.append_row(SRC / "results.csv", row)
                    runner.append_row(AGG_CSV, row)
                    log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                        f"ppl={row.ppl} mean_kld={row.mean_kld} "
                        f"same_top_p={row.same_top_p}")

    # Stage 3: IQ4_NL × 5
    iq4_rows: list[dict] = []
    for cell, imat_name, _imat_ctx, _ds in CELLS:
        imat = SRC / imat_name
        qpath = WORK / f"IQ4_NL-{cell}.gguf"
        with phase(f"IQ4_NL {cell}"):
            step(f"quantize IQ4_NL ({cell})", qpath,
                 lambda q=qpath, s=imat, n=cell: gguf.quantize(
                     F16, q, "IQ4_NL", imatrix=s,
                     log=LOGS / f"quantize-iq4nl-{n}.log"))
            label = f"Jackrong/Qwopus3.5-9B-Coder|IQ4_NL|imatrix|{cell}"
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
    q4km_rows = _existing_rows("Jackrong/Qwopus3.5-9B-Coder|imatrix|")
    md_path = WORK / "table.md"
    lines = [
        "| quant | technique | dataset | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |",
        "|---|---|---|---|---|---|---|---|",
        f"| FP16   | none    | —             | {fp16_size_gib:.2f} | {fp16_bpw:.3f} | "
        f"{fp16_ppl:.4f} | 0.00000 | 100.0000 |",
    ]
    cell_to_ds_label = {c: ds for c, _, _, ds in CELLS}
    table_order = ["wiki", "custom", "mixed512", "mixed2k", "mixed8k"]

    def _fmt(v, places: int) -> str:
        try:
            return f"{float(v):.{places}f}"
        except (TypeError, ValueError):
            return "—"

    for cell in table_order:
        ds = cell_to_ds_label[cell]
        r = q4km_rows.get(ds)
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
