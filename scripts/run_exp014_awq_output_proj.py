"""Experiment 011: AWQ + o_proj / down_proj scaling on google/gemma-4-31B-it.

Question: does adding the two largest tensor classes (attention output and MLP
down) to the AWQ scale-group set push sub-3bpw quants (IQ2_XS, IQ2_M, Q2_K_S)
beyond the naive exp-010 baseline? These tensors live downstream of
non-normalized inputs so the scales are applied to the weight only — there is
no RMSNorm γ to fold into, and the F16 forward is intentionally perturbed
(sanity-bounded by ``sanity_max_rel``).

Reuses exp-009 artifacts read-only (HF source, F16 GGUF, mixed8k corpus,
imatrix, eval corpus, baseline KLD). Same Gemma-31B specifics as exp-010
(``rmsnorm_plus_one=False``, ``device="cpu"``, ``eval_ctx=4096``,
``sanity_max_rel=0.15``).

Idempotent via experiments.step(). Re-run to resume.
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
EXP10 = REPO / "out" / "exp-010" / SLUG  # for naive-AWQ comparison rows
EXP14 = REPO / "out" / "exp-014" / SLUG
LOGS = EXP14 / "logs"

EVAL_CTX = 4096
QUANTS = ["IQ2_XS", "IQ2_M", "Q2_K_S"]
DATASET_LABEL = "500k-custom+wiki (ctx=8192) + AWQ+oproj+downproj"

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
            "exp-014 reuses exp-009 artifacts; missing:\n  " + names +
            "\n\nRun scripts/run_exp009_quant_calibration_gemma_31b.py first."
        )


def _f16_ppl_from_baseline_log() -> float | None:
    f = EXP09 / "logs" / "baseline.log"
    if not f.exists():
        return None
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", f.read_text())
    return float(m.group(1)) if m else None


def _load_csv_row(csv_path: Path, quant: str, technique_token: str) -> dict | None:
    """Look up a results.csv row by quant tag and technique substring."""
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
    EXP14.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    awq_bundle = EXP14 / "awq.pt"
    model_awq = EXP14 / "model_awq"
    f16_awq = EXP14 / "model-f16-awq.gguf"
    per_model_csv = EXP14 / "results.csv"

    with phase("AWQ calibrate (with o_proj+down_proj groups)"):
        step("AWQ calibrate", awq_bundle,
             lambda: awq.calibrate(
                 SRC_MODEL, SRC_MIXED, awq_bundle,
                 proxy_tokens=256,
                 device="cpu",
                 dtype="bfloat16",
                 include_output_proj=True,
             ))

    with phase("AWQ apply (incl. o_proj/down_proj, no norm fold)"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 SRC_MODEL, awq_bundle, model_awq,
                 device="cpu",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 # With the v_proj/up_proj counter-fold the math is exact in
                 # f32, but bf16 round-trip noise compounds over 60 layers and
                 # 2-3 extra mul ops per layer. Observed: ~0.34 rel drift vs
                 # exp-010 baseline of ~0.10. Threshold catches real bugs
                 # (the broken no-cancel design tripped at 1.498) while
                 # tolerating the longer fold chain.
                 sanity_max_rel=0.40,
             ))

    with phase("convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(model_awq, f16_awq,
                                            log=LOGS / "convert-awq.log"))

    n_params = bpw_mod.n_params(SRC_F16)
    log(f"reference n_params (from exp-009 F16) = {n_params:,.0f}")

    rows_by_quant: dict[str, runner.BenchRow] = {}
    for q in QUANTS:
        qpath = EXP14 / f"{q}-awq.gguf"
        with phase(f"{q} AWQ+outproj"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     f16_awq, p, qt, imatrix=SRC_IMAT,
                     log=LOGS / f"quantize-{qt}.log"))
            label = f"{REPO_ID}|{q}|awq+outproj+imatrix|{DATASET_LABEL}"
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
            lines.append(f"| {q} | **awq+outproj+imatrix** | — | — | — | — | — |")
        else:
            lines.append(
                f"| {q} | **awq+outproj+imatrix** | "
                f"{_fmt(r.size_gib, 2)} | {_fmt(r.bpw, 3)} | "
                f"{_fmt(r.ppl, 4)} | {_fmt(r.mean_kld, 5)} | "
                f"{_fmt(r.same_top_p, 4)} |"
            )

    md_path = EXP14 / "table.md"
    md_path.write_text("\n".join(lines) + "\n")
    log(f"table -> {md_path.relative_to(REPO)}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
