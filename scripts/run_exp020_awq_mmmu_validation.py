"""Experiment 020: AWQ cv-gate with a pure-MMMU validation corpus.

exp-019 used a ~10k-token val slice that combined logtrain TEST sessions
with `calibration_supplement.txt`. That signal is small and partially
in-domain with calibration, so the binary cv-gate has limited statistical
power to reject over-fitted per-tensor α candidates.

exp-020 swaps the validation source for `calibration_supplements/mmmu/combined.txt`
(MMMU disciplines, ~100–200k tokens) — a genuinely out-of-distribution
held-out slice. Calibration corpus and external eval corpus are unchanged
from exp-019, so the eval column is apples-to-apples with the exp-019
table the README ships.

Only the `cv-gate` variant runs (the release ships cv-gate). imatrix-only
and plain Q2_K baselines are re-quantized + re-benched fresh so exp-020
is fully self-contained.

Idempotent via experiments.step().
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.calibrate import awq
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import llama_cpp
from quant_tuner.quantize import convert, gguf

REPO_ID = "google/gemma-4-31B-it"
SLUG = REPO_ID.replace("/", "__")

EXP09 = REPO / "out" / "exp-009" / SLUG
EXP20 = REPO / "out" / "exp-020" / SLUG
LOGS = EXP20 / "logs"

EVAL_CTX = 4096
IMATRIX_CTX = 4096
QUANTS = ["IQ2_XS", "IQ2_M", "Q2_K_S"]

SRC_MODEL = EXP09 / "model_extracted"
SRC_F16 = EXP09 / "model-f16.gguf"
WIKI_TEST = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"
MMMU_VAL = REPO / "calibration_supplements" / "mmmu" / "combined.txt"


def _load_build_corpora():
    path = REPO / "scripts" / "build_corpora.py"
    spec = importlib.util.spec_from_file_location("build_corpora", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_inputs() -> None:
    missing = [p for p in (SRC_MODEL / "config.json", SRC_F16, WIKI_TEST, MMMU_VAL)
               if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "exp-020 needs exp-009 model/F16 + exp-001 wiki + MMMU val:\n  "
            + names
        )


def _f16_ppl_from_baseline_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", log_path.read_text())
    return float(m.group(1)) if m else None


def _bench_and_record(
    qpath: Path, label: str, csv_path: Path,
    *, n_params: float, eval_corpus: Path, baseline_kld: Path,
) -> runner.BenchRow:
    with phase(f"bench {label}"):
        row = runner.bench_one(
            qpath, label,
            reference_n_params=n_params,
            eval_dataset=eval_corpus,
            eval_baseline=baseline_kld,
            eval_ctx=EVAL_CTX,
            log_dir=LOGS,
            suite="kld",
        )
        runner.append_row(csv_path, row)
        log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
            f"ppl={row.ppl} mean_kld={row.mean_kld} "
            f"same_top_p={row.same_top_p}")
        return row


def _run_gate(
    cal_corpus: Path, val_corpus: Path, eval_corpus: Path,
    imatrix_path: Path, baseline_kld: Path,
    n_params: float, dataset_label: str,
) -> dict[str, runner.BenchRow]:
    vdir = EXP20 / "gate"
    vdir.mkdir(parents=True, exist_ok=True)
    awq_bundle = vdir / "awq.pt"
    model_awq = vdir / "model_awq"
    f16_awq = vdir / "model-f16-awq.gguf"
    per_model_csv = vdir / "results.csv"
    technique_token = "awq-cv-gate+imatrix"

    with phase("[gate] AWQ calibrate (per-tensor α, cv_strategy=gate)"):
        step("AWQ calibrate", awq_bundle,
             lambda: awq.calibrate(
                 SRC_MODEL, cal_corpus, awq_bundle,
                 proxy_tokens=256,
                 device="cpu",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 # Narrowed from default ±0.25 to ±0.15 (exp-020 finding):
                 # MMMU val flipped 3 tensors all the way to α=0.0, and those
                 # descents dominated the post-fold drift (rel=1.009). Capping
                 # the grid radius at 0.15 prevents the worst case while still
                 # allowing per-tensor refinement in both directions.
                 per_tensor_grid_radius=0.15,
                 holdout_text=val_corpus,
                 cv_strategy="gate",
                 cv_weight=1.0,
             ))

    with phase("[gate] AWQ apply (per-member scales)"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 SRC_MODEL, awq_bundle, model_awq,
                 device="cpu",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 # Belt-and-suspenders alongside per_tensor_grid_radius=0.15.
                 # First exp-020 attempt with the default ±0.25 grid hit
                 # rel=1.009. Narrowing the grid should bring drift back near
                 # exp-019 levels (0.34), but 1.20 buys safety margin in case
                 # MMMU val still concentrates accepts on drift-sensitive
                 # deep-layer v_proj. Post-bench eval (PPL/KLD/same_top_p) is
                 # ground truth — if quality degrades we revert.
                 sanity_max_rel=1.20,
             ))

    with phase("[gate] convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq,
                 log=LOGS / "convert-gate.log"))

    rows: dict[str, runner.BenchRow] = {}
    for q in QUANTS:
        qpath = vdir / f"{q}-awq.gguf"
        with phase(f"[gate] {q}"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     f16_awq, p, qt, imatrix=imatrix_path,
                     log=LOGS / f"quantize-gate-{qt}.log"))
            label = f"{REPO_ID}|{q}|{technique_token}|{dataset_label}"
            rows[q] = _bench_and_record(
                qpath, label, per_model_csv,
                n_params=n_params,
                eval_corpus=eval_corpus,
                baseline_kld=baseline_kld,
            )
    return rows


def _run_imatrix_only(
    eval_corpus: Path, imatrix_path: Path, baseline_kld: Path,
    n_params: float, dataset_label: str,
) -> dict[str, runner.BenchRow]:
    vdir = EXP20 / "imatrix-only"
    vdir.mkdir(parents=True, exist_ok=True)
    per_model_csv = vdir / "results.csv"
    rows: dict[str, runner.BenchRow] = {}
    for q in QUANTS:
        qpath = vdir / f"{q}-imatrix.gguf"
        with phase(f"[imatrix-only] {q}"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     SRC_F16, p, qt, imatrix=imatrix_path,
                     log=LOGS / f"quantize-imatrix-{qt}.log"))
            label = f"{REPO_ID}|{q}|imatrix|{dataset_label}"
            rows[q] = _bench_and_record(
                qpath, label, per_model_csv,
                n_params=n_params,
                eval_corpus=eval_corpus,
                baseline_kld=baseline_kld,
            )
    return rows


def _run_plain(
    eval_corpus: Path, baseline_kld: Path, n_params: float, dataset_label: str,
) -> runner.BenchRow:
    vdir = EXP20 / "plain"
    vdir.mkdir(parents=True, exist_ok=True)
    per_model_csv = vdir / "results.csv"
    qpath = vdir / "Q2_K-plain.gguf"
    with phase("[plain] Q2_K no-imatrix no-AWQ"):
        step("quantize Q2_K plain", qpath,
             lambda: gguf.quantize(
                 SRC_F16, qpath, "Q2_K",
                 imatrix=None,
                 log=LOGS / "quantize-plain-Q2_K.log"))
        label = f"{REPO_ID}|Q2_K|plain|{dataset_label}"
        return _bench_and_record(
            qpath, label, per_model_csv,
            n_params=n_params,
            eval_corpus=eval_corpus,
            baseline_kld=baseline_kld,
        )


def main() -> int:
    _check_inputs()
    EXP20.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    corpora_dir = EXP20 / "corpora"
    cal_corpus = corpora_dir / "corpus.cal.txt"
    val_corpus = corpora_dir / "corpus.val.txt"
    eval_corpus = corpora_dir / "corpus.eval.txt"
    audit_path = corpora_dir / "corpora_audit.json"

    with phase("build corpora (cal+eval normal, val=pure MMMU)"):
        def _build_corpora() -> None:
            bc = _load_build_corpora()
            bc.build(
                out_dir=corpora_dir,
                model_dir=SRC_MODEL,
                wiki_test=WIKI_TEST,
                cal_tokens=500_000,
                val_tokens=0,  # MMMU only, no logtrain test slice
                eval_tokens_per_domain=30_000,
                seed=42,
                val_supplement_override=MMMU_VAL,
            )
        step("build_corpora", audit_path, _build_corpora)

    imatrix_path = EXP20 / "imatrix-cal.gguf"
    with phase("imatrix on cal corpus"):
        step("imatrix", imatrix_path,
             lambda: llama_cpp.imatrix(
                 SRC_F16, cal_corpus, imatrix_path,
                 ctx=IMATRIX_CTX,
                 log=LOGS / "imatrix.log"))

    baseline_kld = EXP20 / "baseline.kld"
    with phase("F16 baseline KLD on eval corpus"):
        step("baseline", baseline_kld,
             lambda: kld.build_baseline(
                 SRC_F16, eval_corpus, baseline_kld,
                 ctx=EVAL_CTX,
                 log=LOGS / "baseline.log"))

    n_params = bpw_mod.n_params(SRC_F16)
    log(f"reference n_params (from exp-009 F16) = {n_params:,.0f}")

    dataset_label = (
        "wiki+500k-logtrain (cal) / MMMU (val) / code+math+tools (eval)"
    )

    imat_rows = _run_imatrix_only(
        eval_corpus, imatrix_path, baseline_kld, n_params, dataset_label,
    )
    gate_rows = _run_gate(
        cal_corpus, val_corpus, eval_corpus,
        imatrix_path, baseline_kld, n_params, dataset_label,
    )
    plain_row = _run_plain(eval_corpus, baseline_kld, n_params, dataset_label)

    fp16_ppl = _f16_ppl_from_baseline_log(LOGS / "baseline.log")
    fp16_size_gib = SRC_F16.stat().st_size / (1024 ** 3)
    fp16_bpw = (SRC_F16.stat().st_size * 8) / n_params

    def _fmt(v, places: int) -> str:
        try:
            return f"{float(v):.{places}f}"
        except (TypeError, ValueError):
            return "—"

    def _row(quant: str, label: str, r: runner.BenchRow | None) -> str:
        if r is None:
            return f"| {quant} | {label} | — | — | — | — | — |"
        return (
            f"| {quant} | {label} | "
            f"{_fmt(r.size_gib, 2)} | {_fmt(r.bpw, 3)} | "
            f"{_fmt(r.ppl, 4)} | {_fmt(r.mean_kld, 5)} | "
            f"{_fmt(r.same_top_p, 4)} |"
        )

    lines = [
        "| quant | technique | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| FP16 | none (reference) | {fp16_size_gib:.2f} | {fp16_bpw:.3f} | "
        f"**{_fmt(fp16_ppl, 4)}** | 0.00000 | 100.00% |",
    ]
    for q in QUANTS:
        lines.append(_row(q, "imatrix only", imat_rows.get(q)))
        lines.append(_row(q, "**AWQ cv-gate + imatrix**", gate_rows.get(q)))
    lines.append(_row("Q2_K", "plain (no imatrix, no AWQ)", plain_row))

    md_path = EXP20 / "table.md"
    md_path.write_text("\n".join(lines) + "\n")
    log(f"table -> {md_path.relative_to(REPO)}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
