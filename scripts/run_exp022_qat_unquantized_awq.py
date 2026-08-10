"""Experiment 022: AWQ cv-gate + imatrix on the QAT-unquantized Gemma.

Tests whether QAT (quantization-aware training, optimized for Q4_0) leaves
weights that are friendlier to *sub-3-bpw* quantization than the vanilla
FP16 base model. Same recipe as exp-020, single input swap:

    SRC = google/gemma-4-31B-it-qat-q4_0-unquantized   (was google/gemma-4-31B-it)

Everything else — corpora, validation slice, eval baseline, AWQ knobs,
quants, sanity gates — is held identical to exp-020 so each exp-022 row
directly pairs with its exp-020 counterpart.

Baseline KLD reuses exp-020's `baseline.kld` (vanilla FP16). All KLD /
same_top_p numbers therefore measure "agreement with the original
google/gemma-4-31B-it FP16 model" — the right metric for asking "does
the QAT-routed quantization land closer to the original model?"

Arms:
  - AWQ cv-gate + imatrix:   IQ2_XS, IQ2_M, Q2_K_S  (from QAT-HF safetensors)
  - imatrix only:            IQ2_XS, IQ2_M, Q2_K_S  (from QAT-F16 GGUF)
  - plain:                   Q2_K                    (from QAT-F16 GGUF)

Idempotent via experiments.step(). Downloads ~62 GiB on first run.
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
from quant_tuner.models import extract, llama_cpp
from quant_tuner.quantize import convert, gguf

VANILLA_REPO_ID = "google/gemma-4-31B-it"
QAT_REPO_ID = "google/gemma-4-31B-it-qat-q4_0-unquantized"
SLUG = QAT_REPO_ID.replace("/", "__")

EXP20 = REPO / "out" / "exp-020" / VANILLA_REPO_ID.replace("/", "__")
EXP22 = REPO / "out" / "exp-022" / SLUG
LOGS = EXP22 / "logs"

EVAL_CTX = 4096
IMATRIX_CTX = 4096
QUANTS = ["IQ2_XS", "IQ2_M", "Q2_K_S"]

SRC_MODEL = EXP22 / "model_extracted"
SRC_F16 = EXP22 / "model-f16.gguf"

# Reuse exp-020 inputs verbatim — corpora are tokenizer-agnostic text and
# the QAT model shares google/gemma-4-31B-it's tokenizer. Baseline KLD is
# the vanilla-FP16 reference so KLD/same_top_p stays comparable to exp-020.
EXP20_CORPORA = EXP20 / "corpora"
EXP20_BASELINE_KLD = EXP20 / "baseline.kld"
EXP20_VANILLA_F16 = REPO / "out" / "exp-009" / VANILLA_REPO_ID.replace("/", "__") / "model-f16.gguf"


def _check_inputs() -> None:
    missing = [
        p for p in (
            EXP20_CORPORA / "corpus.cal.txt",
            EXP20_CORPORA / "corpus.val.txt",
            EXP20_CORPORA / "corpus.eval.txt",
            EXP20_BASELINE_KLD,
            EXP20_VANILLA_F16,
        ) if not p.exists()
    ]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "exp-022 reuses exp-020 corpora + baseline + exp-009 vanilla F16:\n  "
            + names
        )


def _fetch_qat_hf() -> None:
    """Download QAT-unquantized HF model and strip the vision tower.

    The QAT-unquantized repo ships the full multimodal Gemma-4 checkpoint
    (text + vision), so weights live under `model.language_model.*`. AWQ's
    `discover_groups` expects a plain CausalLM with `model.layers.*`, so we
    run `extract_text_lm` to drop the vision prefixes and rename the
    language-model prefix in place.
    """
    SRC_MODEL.parent.mkdir(parents=True, exist_ok=True)
    # If a prior failed run left a symlink here, remove it so extract_text_lm
    # can write a real directory.
    if SRC_MODEL.is_symlink():
        SRC_MODEL.unlink()
    extract.extract_text_lm(
        source=QAT_REPO_ID,
        output_dir=SRC_MODEL,
        keep_mtp=False,
    )


def _f16_ppl_from_log(log_path: Path) -> float | None:
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
    vdir = EXP22 / "gate"
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
            label = f"{QAT_REPO_ID}|{q}|{technique_token}|{dataset_label}"
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
    vdir = EXP22 / "imatrix-only"
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
            label = f"{QAT_REPO_ID}|{q}|imatrix|{dataset_label}"
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
    vdir = EXP22 / "plain"
    vdir.mkdir(parents=True, exist_ok=True)
    per_model_csv = vdir / "results.csv"
    qpath = vdir / "Q2_K-plain.gguf"
    with phase("[plain] Q2_K no-imatrix no-AWQ"):
        step("quantize Q2_K plain", qpath,
             lambda: gguf.quantize(
                 SRC_F16, qpath, "Q2_K",
                 imatrix=None,
                 log=LOGS / "quantize-plain-Q2_K.log"))
        label = f"{QAT_REPO_ID}|Q2_K|plain|{dataset_label}"
        return _bench_and_record(
            qpath, label, per_model_csv,
            n_params=n_params,
            eval_corpus=eval_corpus,
            baseline_kld=baseline_kld,
        )


def main() -> int:
    _check_inputs()
    EXP22.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    with phase("fetch QAT-unquantized HF model"):
        step("snapshot_download", SRC_MODEL / "config.json", _fetch_qat_hf)

    with phase("convert QAT-HF -> F16 GGUF"):
        step("convert", SRC_F16,
             lambda: convert.hf_to_f16_gguf(
                 SRC_MODEL, SRC_F16,
                 log=LOGS / "convert-f16.log"))

    cal_corpus = EXP20_CORPORA / "corpus.cal.txt"
    val_corpus = EXP20_CORPORA / "corpus.val.txt"
    eval_corpus = EXP20_CORPORA / "corpus.eval.txt"

    imatrix_path = EXP22 / "imatrix-cal.gguf"
    with phase("imatrix on cal corpus (QAT-F16)"):
        step("imatrix", imatrix_path,
             lambda: llama_cpp.imatrix(
                 SRC_F16, cal_corpus, imatrix_path,
                 ctx=IMATRIX_CTX,
                 log=LOGS / "imatrix.log"))

    # Use vanilla-FP16 baseline so KLD measures drift from the *original*
    # google/gemma-4-31B-it. n_params likewise derived from the vanilla F16
    # so BPW is comparable to exp-020.
    n_params = bpw_mod.n_params(EXP20_VANILLA_F16)
    log(f"reference n_params (from vanilla FP16) = {n_params:,.0f}")

    dataset_label = (
        "wiki+500k-logtrain (cal) / MMMU (val) / code+math+tools (eval) "
        "// baseline=vanilla-FP16"
    )

    imat_rows = _run_imatrix_only(
        eval_corpus, imatrix_path, EXP20_BASELINE_KLD, n_params, dataset_label,
    )
    gate_rows = _run_gate(
        cal_corpus, val_corpus, eval_corpus,
        imatrix_path, EXP20_BASELINE_KLD, n_params, dataset_label,
    )
    plain_row = _run_plain(
        eval_corpus, EXP20_BASELINE_KLD, n_params, dataset_label,
    )

    fp16_ppl = _f16_ppl_from_log(EXP20 / "logs" / "baseline.log")

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

    fp16_size_gib = EXP20_VANILLA_F16.stat().st_size / (1024 ** 3)
    fp16_bpw = (EXP20_VANILLA_F16.stat().st_size * 8) / n_params

    lines = [
        "| quant | technique (QAT-source) | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| FP16 | vanilla-FP16 (reference) | {fp16_size_gib:.2f} | {fp16_bpw:.3f} | "
        f"**{_fmt(fp16_ppl, 4)}** | 0.00000 | 100.00% |",
    ]
    for q in QUANTS:
        lines.append(_row(q, "imatrix only (QAT-F16)", imat_rows.get(q)))
        lines.append(_row(q, "**AWQ cv-gate + imatrix (QAT-HF)**", gate_rows.get(q)))
    lines.append(_row("Q2_K", "plain (QAT-F16, no imatrix, no AWQ)", plain_row))

    md_path = EXP22 / "table.md"
    md_path.write_text("\n".join(lines) + "\n")
    log(f"table -> {md_path.relative_to(REPO)}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
