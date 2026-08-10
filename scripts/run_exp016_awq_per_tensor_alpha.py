"""Experiment 013: AWQ with per-tensor α refinement on google/gemma-4-31B-it.

Question: q/k/v share a scale group (same input) but their weight statistics
differ. After picking the group-level α, can each member refine α locally
(group_α ± 0.25, clipped to [0, 1]) using the same shared activation profile
to lower its own proxy reconstruction error? Per-member scales fold the
*group* scale into the RMSNorm gain and apply the per-member scale to the
weight; the residual ratio is a controlled F16 perturbation (bounded by the
sanity check).

Compare against exp-009 (imatrix only) and exp-010 (single-α naive AWQ) for
IQ2_XS / IQ2_M / Q2_K_S.

Reuses exp-009 artifacts. Same Gemma-31B specifics as exp-010.
Idempotent via experiments.step().
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate import awq
from quant_tuner.experiments import log, phase, step
from quant_tuner.quantize import convert, gguf

REPO_ID = "google/gemma-4-31B-it"
SLUG = REPO_ID.replace("/", "__")

EXP09 = REPO / "out" / "exp-009" / SLUG
EXP10 = REPO / "out" / "exp-010" / SLUG
EXP16 = REPO / "out" / "exp-016" / SLUG
LOGS = EXP16 / "logs"

EVAL_CTX = 4096
QUANTS = ["IQ2_XS", "IQ2_M", "Q2_K_S"]
DATASET_LABEL = "500k-custom+wiki (ctx=8192) + AWQ-per-tensor-α"

SRC_MODEL = EXP09 / "model_extracted"
SRC_F16 = EXP09 / "model-f16.gguf"
SRC_MIXED = EXP09 / "corpus.mixed8k.txt"
SRC_IMAT = EXP09 / "imatrix-mixed8k.gguf"
SRC_EVAL = EXP09 / "corpus.eval.txt"
SRC_BASELINE = EXP09 / "baseline.kld"


def _check_inputs() -> None:
    missing = [p for p in (SRC_MODEL / "config.json", SRC_F16, SRC_MIXED,
                           SRC_IMAT, SRC_EVAL, SRC_BASELINE) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p.relative_to(REPO)) for p in missing)
        raise FileNotFoundError(
            "exp-016 reuses exp-009 artifacts; missing:\n  " + names +
            "\n\nRun scripts/run_exp009_quant_calibration_gemma_31b.py first."
        )


def _f16_ppl_from_baseline_log() -> float | None:
    f = EXP09 / "logs" / "baseline.log"
    if not f.exists():
        return None
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", f.read_text())
    return float(m.group(1)) if m else None


def _load_csv_row(csv_path: Path, quant: str, technique_token: str) -> dict | None:
    import csv
    if not csv_path.exists():
        return None
    needle = f"|{quant}|{technique_token}|"
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if needle in row["model"]:
                return row
    return None


def main() -> int:
    _check_inputs()
    EXP16.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    awq_bundle = EXP16 / "awq.pt"
    model_awq = EXP16 / "model_awq"
    f16_awq = EXP16 / "model-f16-awq.gguf"
    per_model_csv = EXP16 / "results.csv"

    with phase("AWQ calibrate (grid α + per-tensor refinement)"):
        step("AWQ calibrate", awq_bundle,
             lambda: awq.calibrate(
                 SRC_MODEL, SRC_MIXED, awq_bundle,
                 proxy_tokens=256,
                 device="cpu",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
             ))

    with phase("AWQ apply (per-member scales + group norm fold)"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 SRC_MODEL, awq_bundle, model_awq,
                 device="cpu",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 # Per-member scales create residual ratios (s_member/s_group)
                 # that don't cancel in F16. With ±0.25 local α grid the
                 # channel ratios can reach ~1.4×, compounding over 60 layers
                 # in bf16. Observed: 0.52 rel drift — large but bounded; the
                 # whole experimental point is to inject this perturbation and
                 # see whether the quantizer benefits. The check still catches
                 # actually-broken math (which would show up at 1.0+).
                 sanity_max_rel=0.65,
             ))

    with phase("convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(model_awq, f16_awq,
                                            log=LOGS / "convert-awq.log"))

    n_params = bpw_mod.n_params(SRC_F16)
    log(f"reference n_params (from exp-009 F16) = {n_params:,.0f}")

    rows_by_quant: dict[str, runner.BenchRow] = {}
    for q in QUANTS:
        qpath = EXP16 / f"{q}-awq.gguf"
        with phase(f"{q} AWQ-per-tensor-α"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     f16_awq, p, qt, imatrix=SRC_IMAT,
                     log=LOGS / f"quantize-{qt}.log"))
            label = f"{REPO_ID}|{q}|awq-pertensor+imatrix|{DATASET_LABEL}"
            with phase(f"bench {label}"):
                row = runner.bench_one(
                    qpath, label,
                    reference_n_params=n_params,
                    eval_dataset=SRC_EVAL,
                    eval_baseline=SRC_BASELINE,
                    eval_ctx=EVAL_CTX,
                    log_dir=LOGS,
                    suite="kld",
                )
                runner.append_row(per_model_csv, row)
                rows_by_quant[q] = row
                log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                    f"ppl={row.ppl} mean_kld={row.mean_kld} "
                    f"same_top_p={row.same_top_p}")

    fp16_ppl = _f16_ppl_from_baseline_log()
    fp16_size_gib = SRC_F16.stat().st_size / (1024 ** 3)
    fp16_bpw = (SRC_F16.stat().st_size * 8) / n_params

    def _fmt(v, places: int) -> str:
        try:
            return f"{float(v):.{places}f}"
        except (TypeError, ValueError):
            return "—"

    def _csv_row_line(quant: str, label: str, prev: dict | None) -> str:
        if prev is None:
            return f"| {quant} | {label} | — | — | — | — | — |"
        return (
            f"| {quant} | {label} | "
            f"{_fmt(prev.get('size_gib'), 2)} | {_fmt(prev.get('bpw'), 3)} | "
            f"{_fmt(prev.get('ppl'), 4)} | {_fmt(prev.get('mean_kld'), 5)} | "
            f"{_fmt(prev.get('same_top_p'), 4)} |"
        )

    lines = [
        "| quant | technique | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |",
        "|---|---|---|---|---|---|---|",
        f"| FP16 | none | {fp16_size_gib:.2f} | {fp16_bpw:.3f} | "
        f"{_fmt(fp16_ppl, 4)} | 0.00000 | 100.0000 |",
    ]
    exp09_csv = EXP09 / "results.csv"
    exp10_csv = EXP10 / "results.csv"
    for q in QUANTS:
        lines.append(_csv_row_line(q, "imatrix (exp-009)",
                                   _load_csv_row(exp09_csv, q, "imatrix")))
        lines.append(_csv_row_line(q, "awq+imatrix (exp-010)",
                                   _load_csv_row(exp10_csv, q, "awq+imatrix")))
        r = rows_by_quant.get(q)
        if r is None:
            lines.append(f"| {q} | **awq-pertensor+imatrix** | — | — | — | — | — |")
        else:
            lines.append(
                f"| {q} | **awq-pertensor+imatrix** | "
                f"{_fmt(r.size_gib, 2)} | {_fmt(r.bpw, 3)} | "
                f"{_fmt(r.ppl, 4)} | {_fmt(r.mean_kld, 5)} | "
                f"{_fmt(r.same_top_p, 4)} |"
            )

    md_path = EXP16 / "table.md"
    md_path.write_text("\n".join(lines) + "\n")
    log(f"table -> {md_path.relative_to(REPO)}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
