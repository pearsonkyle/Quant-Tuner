"""Experiment 019: AWQ cv-mixed / cv-gate on disjoint corpora.

exp-017 (cv-gate) and exp-018 (cv-mixed) drew cal and held-out from the
same logtrain distribution, so the held-out term measured sampling noise,
not generalization. exp-019 wires the same scoring into the disjoint
corpora produced by `scripts/build_corpora.py`:

    cal  = ALL wiki.test.raw + ~500K logtrain TRAIN tokens
    val  = ~10K logtrain TEST tokens + calibration_supplement.txt
    eval = external eaddario/imatrix-calibration {code,math,tools}_small

Calibration logic in `awq.calibrate` is unchanged. Both cv_strategy
variants ("mixed" with cv_weight=2.0, and "gate") run here so the
comparison is apples-to-apples on identical corpora.

Reuses exp-009's extracted HF model + F16 GGUF. Rebuilds imatrix and the
KLD baseline because the cal and eval corpora both change.

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
EXP10 = REPO / "out" / "exp-010" / SLUG
EXP17 = REPO / "out" / "exp-017" / SLUG
EXP18 = REPO / "out" / "exp-018" / SLUG
EXP19 = REPO / "out" / "exp-019" / SLUG
LOGS = EXP19 / "logs"

EVAL_CTX = 4096
IMATRIX_CTX = 512
QUANTS = ["IQ2_XS", "IQ2_M", "Q2_K_S"]

SRC_MODEL = EXP09 / "model_extracted"
SRC_F16 = EXP09 / "model-f16.gguf"
WIKI_TEST = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"


def _load_build_corpora():
    """Import scripts/build_corpora.py without making `scripts/` a package."""
    path = REPO / "scripts" / "build_corpora.py"
    spec = importlib.util.spec_from_file_location("build_corpora", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_inputs() -> None:
    missing = [p for p in (SRC_MODEL / "config.json", SRC_F16, WIKI_TEST)
               if not p.exists()]
    if missing:
        names = "\n  ".join(str(p.relative_to(REPO)) for p in missing)
        raise FileNotFoundError(
            "exp-019 needs exp-009 model/F16 + exp-001 wiki:\n  " + names +
            "\n\nRun the upstream experiments first."
        )


def _f16_ppl_from_baseline_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", log_path.read_text())
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


def _run_variant(
    variant: str,
    *,
    cv_strategy: str,
    cv_weight: float,
    cal_corpus: Path,
    val_corpus: Path,
    eval_corpus: Path,
    imatrix_path: Path,
    baseline_kld: Path,
    n_params: float,
    dataset_label: str,
) -> dict[str, runner.BenchRow]:
    """Calibrate + apply + convert + quantize + bench for one cv_strategy."""
    vdir = EXP19 / variant
    vdir.mkdir(parents=True, exist_ok=True)

    awq_bundle = vdir / "awq.pt"
    model_awq = vdir / "model_awq"
    f16_awq = vdir / "model-f16-awq.gguf"
    per_model_csv = vdir / "results.csv"

    technique_token = f"awq-cv-{variant}+imatrix"

    with phase(f"[{variant}] AWQ calibrate (per-tensor α, cv_strategy={cv_strategy})"):
        step("AWQ calibrate", awq_bundle,
             lambda: awq.calibrate(
                 SRC_MODEL, cal_corpus, awq_bundle,
                 proxy_tokens=256,
                 device="cpu",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 holdout_text=val_corpus,
                 cv_strategy=cv_strategy,
                 cv_weight=cv_weight,
             ))

    with phase(f"[{variant}] AWQ apply (per-member scales)"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 SRC_MODEL, awq_bundle, model_awq,
                 device="cpu",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 # Matches exp-018 threshold; the val-data swap may move max_rel
                 # either direction. If this trips, capture the value before
                 # raising — it's itself a finding about how disjoint val data
                 # shifts α selection.
                 sanity_max_rel=0.85,
             ))

    with phase(f"[{variant}] convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq,
                 log=LOGS / f"convert-{variant}.log"))

    rows_by_quant: dict[str, runner.BenchRow] = {}
    for q in QUANTS:
        qpath = vdir / f"{q}-awq.gguf"
        with phase(f"[{variant}] {q}"):
            step(f"quantize {q}", qpath,
                 lambda p=qpath, qt=q: gguf.quantize(
                     f16_awq, p, qt, imatrix=imatrix_path,
                     log=LOGS / f"quantize-{variant}-{qt}.log"))
            label = f"{REPO_ID}|{q}|{technique_token}|{dataset_label}"
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
                runner.append_row(per_model_csv, row)
                rows_by_quant[q] = row
                log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                    f"ppl={row.ppl} mean_kld={row.mean_kld} "
                    f"same_top_p={row.same_top_p}")
    return rows_by_quant


def main() -> int:
    _check_inputs()
    EXP19.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    corpora_dir = EXP19 / "corpora"
    cal_corpus = corpora_dir / "corpus.cal.txt"
    val_corpus = corpora_dir / "corpus.val.txt"
    eval_corpus = corpora_dir / "corpus.eval.txt"
    audit_path = corpora_dir / "corpora_audit.json"

    with phase("build disjoint corpora (cal/val/eval)"):
        # Audit file is written last by build_corpora.build(), so use it as
        # the idempotency sentinel: present iff the full build succeeded.
        def _build_corpora() -> None:
            bc = _load_build_corpora()
            bc.build(
                out_dir=corpora_dir,
                model_dir=SRC_MODEL,
                wiki_test=WIKI_TEST,
                cal_tokens=500_000,
                val_tokens=10_000,
                eval_tokens_per_domain=30_000,
                seed=42,
            )
        step("build_corpora", audit_path, _build_corpora)

    imatrix_path = EXP19 / "imatrix-cal.gguf"
    with phase("imatrix on new cal corpus"):
        step("imatrix", imatrix_path,
             lambda: llama_cpp.imatrix(
                 SRC_F16, cal_corpus, imatrix_path,
                 ctx=IMATRIX_CTX,
                 log=LOGS / "imatrix.log"))

    baseline_kld = EXP19 / "baseline.kld"
    with phase("F16 baseline KLD on new eval corpus"):
        step("baseline", baseline_kld,
             lambda: kld.build_baseline(
                 SRC_F16, eval_corpus, baseline_kld,
                 ctx=EVAL_CTX,
                 log=LOGS / "baseline.log"))

    n_params = bpw_mod.n_params(SRC_F16)
    log(f"reference n_params (from exp-009 F16) = {n_params:,.0f}")

    dataset_label = "wiki+500k-logtrain (cal) / 10k-logtrain+supplement (val) / code+math+tools (eval)"

    mixed_rows = _run_variant(
        "mixed",
        cv_strategy="mixed",
        cv_weight=2.0,
        cal_corpus=cal_corpus,
        val_corpus=val_corpus,
        eval_corpus=eval_corpus,
        imatrix_path=imatrix_path,
        baseline_kld=baseline_kld,
        n_params=n_params,
        dataset_label=dataset_label,
    )
    gate_rows = _run_variant(
        "gate",
        cv_strategy="gate",
        cv_weight=1.0,  # unused when strategy="gate"
        cal_corpus=cal_corpus,
        val_corpus=val_corpus,
        eval_corpus=eval_corpus,
        imatrix_path=imatrix_path,
        baseline_kld=baseline_kld,
        n_params=n_params,
        dataset_label=dataset_label,
    )

    # Plain Q2_K: no AWQ, no imatrix. Q2_K (not Q2_K_S) is the most aggressive
    # k-quant llama.cpp will produce without an imatrix — Q2_K_S forces
    # additional tensors into the q2_K block format which hard-requires
    # imatrix data (llama-quant.cpp:779-782). Anchors how much of the
    # calibrated quants' quality comes from calibration vs. the quant format
    # itself. Different BPW (~3.3 vs 2.86 for Q2_K_S), so compare with that
    # caveat in mind.
    plain_dir = EXP19 / "plain"
    plain_dir.mkdir(parents=True, exist_ok=True)
    plain_csv = plain_dir / "results.csv"
    plain_qpath = plain_dir / "Q2_K-plain.gguf"
    plain_row: runner.BenchRow | None = None
    with phase("[plain] Q2_K no-imatrix no-AWQ"):
        step("quantize Q2_K plain", plain_qpath,
             lambda: gguf.quantize(
                 SRC_F16, plain_qpath, "Q2_K",
                 imatrix=None,
                 log=LOGS / "quantize-plain-Q2_K.log"))
        plain_label = (
            f"{REPO_ID}|Q2_K|plain|{dataset_label}"
        )
        with phase(f"bench {plain_label}"):
            plain_row = runner.bench_one(
                plain_qpath, plain_label,
                reference_n_params=n_params,
                eval_dataset=eval_corpus,
                eval_baseline=baseline_kld,
                eval_ctx=EVAL_CTX,
                log_dir=LOGS,
                suite="kld",
            )
            runner.append_row(plain_csv, plain_row)
            log(f"  size={plain_row.size_gib:.2f} GiB bpw={plain_row.bpw:.3f} "
                f"ppl={plain_row.ppl} mean_kld={plain_row.mean_kld} "
                f"same_top_p={plain_row.same_top_p}")

    fp16_ppl = _f16_ppl_from_baseline_log(LOGS / "baseline.log")
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

    def _row_line(quant: str, label: str, r: runner.BenchRow | None) -> str:
        if r is None:
            return f"| {quant} | {label} | — | — | — | — | — |"
        return (
            f"| {quant} | {label} | "
            f"{_fmt(r.size_gib, 2)} | {_fmt(r.bpw, 3)} | "
            f"{_fmt(r.ppl, 4)} | {_fmt(r.mean_kld, 5)} | "
            f"{_fmt(r.same_top_p, 4)} |"
        )

    lines = [
        "**Note:** exp-019 PPL/KLD are measured against the new external eval "
        "corpus (code+math+tools), so they are NOT directly comparable to "
        "exp-009/010/017/018 numbers, which used the logtrain+wiki-derived "
        "eval corpus. Rows from prior experiments are included as historical "
        "context only.",
        "",
        "| quant | technique | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |",
        "|---|---|---|---|---|---|---|",
        f"| FP16 | none | {fp16_size_gib:.2f} | {fp16_bpw:.3f} | "
        f"{_fmt(fp16_ppl, 4)} | 0.00000 | 100.0000 |",
    ]
    exp09_csv = EXP09 / "results.csv"
    exp10_csv = EXP10 / "results.csv"
    exp17_csv = EXP17 / "results.csv"
    exp18_csv = EXP18 / "results.csv"
    for q in QUANTS:
        lines.append(_csv_row_line(q, "imatrix (exp-009)",
                                   _load_csv_row(exp09_csv, q, "imatrix")))
        lines.append(_csv_row_line(q, "awq+imatrix (exp-010)",
                                   _load_csv_row(exp10_csv, q, "awq+imatrix")))
        lines.append(_csv_row_line(q, "awq-cv-gate (exp-017, old corpora)",
                                   _load_csv_row(exp17_csv, q, "awq-cv-gate+imatrix")))
        lines.append(_csv_row_line(q, "awq-cv-mixed (exp-018, old corpora)",
                                   _load_csv_row(exp18_csv, q, "awq-cv-mixed+imatrix")))
        lines.append(_row_line(q, "**awq-cv-gate (exp-019, clean corpora)**",
                               gate_rows.get(q)))
        lines.append(_row_line(q, "**awq-cv-mixed (exp-019, clean corpora)**",
                               mixed_rows.get(q)))

    # Plain Q2_K anchor (different BPW from the QUANTS triple).
    lines.append(
        _row_line("Q2_K", "*plain (exp-019, no imatrix, no AWQ)*", plain_row)
    )
    lines.append("")
    lines.append(
        "*Note: Q2_K (plain) is the most aggressive k-quant llama.cpp will "
        "produce without an imatrix — Q2_K_S forces extra tensors into the "
        "q2_K block format which hard-requires imatrix data. BPW differs "
        "(~3.3 vs 2.86 for Q2_K_S), so use this row as a 'zero calibration "
        "signal' anchor, not a same-bpw comparison.*"
    )

    md_path = EXP19 / "table.md"
    md_path.write_text("\n".join(lines) + "\n")
    log(f"table -> {md_path.relative_to(REPO)}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
