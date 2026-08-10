"""Experiment 010: AWQ → sub-3bpw quant sweep on google/gemma-4-31B-it.

Question: does AWQ rescue Q2_K (and friends) on Gemma-31b? exp-009 ran
the same quants with imatrix only; here we re-quantize from an
AWQ-folded F16 using the **same imatrix and the same FP16 KLD baseline**
so rows are directly comparable.

Quants swept: Q2_K, Q2_K_S, IQ2_M, IQ2_XS.

Reuses exp-009 artifacts read-only (do not delete out/exp-009 before
running this):
  - model_extracted/         HF source
  - model-f16.gguf           FP16 reference
  - corpus.mixed8k.txt       calibration corpus
  - imatrix-mixed8k.gguf     base imatrix for the final llama-quantize pass
  - corpus.eval.txt          held-out eval
  - baseline.kld             FP16 KLD reference (so exp-010 rows compare 1:1 to exp-009)

Gemma specifics (vs the default q4_k_m_awq recipe):
  - rmsnorm_plus_one=False   Gemma is Llama-style γ·x, not Qwen3.5 (1+γ)·x
  - device="cpu" for AWQ apply (31B in bf16 + Metal overhead won't fit)
  - eval_ctx=4096            huge vocab busts llama-perplexity at 8192

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
EXP10 = REPO / "out" / "exp-010" / SLUG
LOGS = EXP10 / "logs"

EVAL_CTX = 4096
QUANTS = ["Q2_K", "Q2_K_S", "IQ2_M", "IQ2_XS"]
DATASET_LABEL = "500k-custom+wiki (ctx=8192) + AWQ"

# Inputs reused from exp-009 (read-only).
SRC_MODEL = EXP09 / "model_extracted"
SRC_F16 = EXP09 / "model-f16.gguf"
SRC_MIXED = EXP09 / "corpus.mixed8k.txt"
SRC_IMAT = EXP09 / "imatrix-mixed8k.gguf"
SRC_EVAL = EXP09 / "corpus.eval.txt"
SRC_BASELINE = EXP09 / "baseline.kld"


def _check_exp009() -> None:
    missing = [p for p in (SRC_MODEL / "config.json", SRC_F16, SRC_MIXED,
                           SRC_IMAT, SRC_EVAL, SRC_BASELINE) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p.relative_to(REPO)) for p in missing)
        raise FileNotFoundError(
            "exp-010 reuses exp-009 artifacts; missing:\n  " + names +
            "\n\nRun scripts/run_exp009_quant_calibration_gemma_31b.py first."
        )


def _f16_ppl_from_baseline_log() -> float | None:
    f = EXP09 / "logs" / "baseline.log"
    if not f.exists():
        return None
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", f.read_text())
    return float(m.group(1)) if m else None


def _load_exp009_row(quant: str) -> dict | None:
    """Pull the matching exp-009 imatrix-only row for side-by-side comparison."""
    import csv
    src = EXP09 / "results.csv"
    if not src.exists():
        return None
    needle = f"|{quant}|imatrix|"
    with open(src) as f:
        for row in csv.DictReader(f):
            if needle in row["model"]:
                return row
    return None


def main() -> int:
    _check_exp009()
    EXP10.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    awq_bundle = EXP10 / "awq.pt"
    model_awq = EXP10 / "model_awq"
    f16_awq = EXP10 / "model-f16-awq.gguf"
    per_model_csv = EXP10 / "results.csv"

    with phase("AWQ calibrate (Gemma-31b mixed8k corpus, α grid)"):
        step("AWQ calibrate (capture mean|x| + grid α)", awq_bundle,
             lambda: awq.calibrate(
                 SRC_MODEL, SRC_MIXED, awq_bundle,
                 # default α grid (0, 0.25, 0.5, 0.75, 1.0) = `best` variant
                 proxy_tokens=256,
                 device="cpu",       # 31B won't fit on MPS with bf16 + Metal overhead
                 dtype="bfloat16",
             ))

    with phase("AWQ apply (fold scales, Llama-style RMSNorm)"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 SRC_MODEL, awq_bundle, model_awq,
                 device="cpu",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,  # Gemma is γ·x, NOT (1+γ)·x
                 # 60 transformer blocks at bf16 compound to ~0.10 rel drift even
                 # after the per-layer algebra is verified correct (12× drop after
                 # the pre_feedforward_layernorm fix). 0.15 catches real bugs while
                 # tolerating bf16 noise at this depth.
                 sanity_max_rel=0.15,
             ))

    with phase("convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(model_awq, f16_awq,
                                            log=LOGS / "convert-awq.log"))

    n_params = bpw_mod.n_params(SRC_F16)
    log(f"reference n_params (from exp-009 F16) = {n_params:,.0f}")

    rows_by_quant: dict[str, runner.BenchRow] = {}
    for q in QUANTS:
        qpath = EXP10 / f"{q}-awq.gguf"
        with phase(f"{q} AWQ"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     f16_awq, p, qt, imatrix=SRC_IMAT,
                     log=LOGS / f"quantize-{qt}.log"))
            label = f"{REPO_ID}|{q}|awq+imatrix|{DATASET_LABEL}"
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

    # Comparison table: AWQ vs imatrix-only (exp-009) for each quant.
    fp16_ppl = _f16_ppl_from_baseline_log()
    fp16_size_gib = SRC_F16.stat().st_size / (1024 ** 3)
    fp16_bpw = (SRC_F16.stat().st_size * 8) / n_params

    def _fmt(v, places: int) -> str:
        try:
            return f"{float(v):.{places}f}"
        except (TypeError, ValueError):
            return "—"

    lines = [
        "| quant | technique | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |",
        "|---|---|---|---|---|---|---|",
        f"| FP16 | none | {fp16_size_gib:.2f} | {fp16_bpw:.3f} | "
        f"{_fmt(fp16_ppl, 4)} | 0.00000 | 100.0000 |",
    ]
    for q in QUANTS:
        prev = _load_exp009_row(q)
        if prev is not None:
            lines.append(
                f"| {q} | imatrix (exp-009) | "
                f"{_fmt(prev.get('size_gib'), 2)} | {_fmt(prev.get('bpw'), 3)} | "
                f"{_fmt(prev.get('ppl'), 4)} | {_fmt(prev.get('mean_kld'), 5)} | "
                f"{_fmt(prev.get('same_top_p'), 4)} |"
            )
        r = rows_by_quant.get(q)
        if r is None:
            lines.append(f"| {q} | awq+imatrix | — | — | — | — | — |")
        else:
            lines.append(
                f"| {q} | **awq+imatrix** | "
                f"{_fmt(r.size_gib, 2)} | {_fmt(r.bpw, 3)} | "
                f"{_fmt(r.ppl, 4)} | {_fmt(r.mean_kld, 5)} | "
                f"{_fmt(r.same_top_p, 4)} |"
            )

    md_path = EXP10 / "table.md"
    md_path.write_text("\n".join(lines) + "\n")
    log(f"table -> {md_path.relative_to(REPO)}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
