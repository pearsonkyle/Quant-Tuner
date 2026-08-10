"""Experiment 040: 2-bit AWQ release for Jackrong/Qwopus3.5-27B-v3 (vanilla only).

Self-contained re-quant mirroring the vanilla half of the gemma-4-31B release
(uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/README.md) on a new model.
There is **no QAT checkpoint** for Qwopus3.5, so only the seven vanilla rows
run — the QAT arm and the QAT-IQ2_M-collapse caveat are dropped entirely.

Settings carried over from the gemma release (exp-020/034/035/038):
  * AWQ proxy_tokens = 1024, ctx = 4096
  * cv_strategy      = gate, cv_weight = 1.0
  * per_tensor_alpha = True, per_tensor_grid_radius = 0.15
  * rmsnorm_plus_one = True (Qwen3.5 uses the (1+γ) RMSNorm — plus_one=False
    collapsed every 2-bit row in the first run; see exp-040b), sanity_max_rel 0.60
  * IQ2_XS AWQ       = auto proxy (iq2_xs codebook), mix = None
  * IQ2_M  AWQ       = proxy q2k_b16 + proxy_mix=IQ2_M (routes the IQ3_S-bumped
                       v_proj/o_proj/down-first-eighth through the iq3_s proxy)
  * Q2_K_S AWQ       = proxy q2k_super (Q2_K super-block proxy), mix = None
  * imatrix          = hybrid, collected on the cal corpus at ctx=4096
  * val (cv-gate)    = pure MMMU (calibration_supplements/mmmu/combined.txt)

7 rows:
  1. Q2_K     plain (no imatrix, no AWQ)
  2. IQ2_XS   imatrix only
  3. IQ2_XS   AWQ cv-gate + imatrix
  4. IQ2_M    imatrix only
  5. IQ2_M    AWQ cv-gate + imatrix (proxy=q2k_b16, mix=IQ2_M)
  6. Q2_K_S   imatrix only
  7. Q2_K_S   AWQ cv-gate + imatrix (proxy=q2k_super)

Phase 0 fetches the HF model, converts to F16 GGUF, builds the three corpora,
collects the imatrix, and measures the F16 baseline KLD. Each AWQ row
auto-deletes its ~2x-model-size intermediates after its bench row lands.
Bench rows dedup against the CSV so a re-run resumes where it died.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp040_release_jackrong27b.py

Writes results to out/exp-040/results.csv; copy the winning GGUFs into
uploads/pearsonkyle/Qwopus3.5-27B-v3-awq-2bit-GGUF/ and fill the README table.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.calibrate import awq
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract, llama_cpp
from quant_tuner.quantize import convert, gguf

# ---- inputs ----------------------------------------------------------------

MODEL_ID = "Jackrong/Qwopus3.5-27B-v3"
MODEL_STEM = "Qwopus3.5-27B-v3"

EXP40 = REPO / "out" / "exp-040"
LOGS = EXP40 / "logs"

HF_DIR = EXP40 / "model_extracted"
F16_GGUF = EXP40 / "model-f16.gguf"
IMATRIX = EXP40 / "imatrix-cal.gguf"
BASELINE_KLD = EXP40 / "baseline.kld"

CORPORA = EXP40 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
VAL_CORPUS = CORPORA / "corpus.val.txt"
EVAL_CORPUS = CORPORA / "corpus.eval.txt"
CORPORA_AUDIT = CORPORA / "corpora_audit.json"

WIKI_TEST = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"
MMMU_VAL = REPO / "calibration_supplements" / "mmmu" / "combined.txt"

PROXY_TOKENS = 1024
CTX = 4096
EVAL_CTX = 4096
IMATRIX_CTX = 4096


# ---- table -----------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    label: str                    # short tag for paths
    quant: str                    # IQ2_XS, IQ2_M, Q2_K_S, Q2_K
    method: str                   # "plain", "imatrix", "awq"
    proxy: str | None = None      # AWQ proxy override; None = auto from quant
    proxy_mix: str | None = None  # AWQ per-member mix override

    @property
    def use_imatrix(self) -> bool:
        return self.method in ("imatrix", "awq")

    @property
    def gguf_name(self) -> str:
        stem = f"{MODEL_STEM}-{self.quant}"
        if self.method == "plain":
            return f"{stem}-plain.gguf"
        if self.method == "imatrix":
            return f"{stem}-imatrix.gguf"
        return f"{stem}-awq-cv-gate.gguf"


ROWS: tuple[Row, ...] = (
    Row("q2k_plain", "Q2_K",   "plain"),
    Row("iq2xs_im",  "IQ2_XS", "imatrix"),
    Row("iq2xs_awq", "IQ2_XS", "awq"),
    Row("iq2m_im",   "IQ2_M",  "imatrix"),
    Row("iq2m_awq",  "IQ2_M",  "awq", proxy="q2k_b16", proxy_mix="IQ2_M"),
    Row("q2ks_im",   "Q2_K_S", "imatrix"),
    Row("q2ks_awq",  "Q2_K_S", "awq", proxy="q2k_super"),
)


# ---------------------------------------------------------------------------

def _load_build_corpora():
    path = REPO / "scripts" / "build_corpora.py"
    spec = importlib.util.spec_from_file_location("build_corpora", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_inputs() -> None:
    missing = [str(p) for p in (WIKI_TEST, MMMU_VAL) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "exp-040 needs exp-001 wiki + MMMU val:\n  " + "\n  ".join(missing)
        )


def _csv_has_qpath(csv_path: Path, qpath: Path) -> bool:
    if not csv_path.exists():
        return False
    needle = str(qpath)
    with csv_path.open() as fh:
        return any(needle in line for line in fh)


def _bench(qpath: Path, label: str, n_params: int, csv_path: Path,
           log_dir: Path) -> None:
    if _csv_has_qpath(csv_path, qpath):
        log(f"  {qpath.name}: bench row already in CSV — skipping")
        return
    row = runner.bench_one(
        qpath, label,
        reference_n_params=n_params,
        eval_dataset=EVAL_CORPUS,
        eval_baseline=BASELINE_KLD,
        eval_ctx=EVAL_CTX,
        log_dir=log_dir,
        suite="kld",
    )
    runner.append_row(csv_path, row)
    log(f"  {qpath.name}: PPL={row.ppl:.4f} mean_KLD={row.mean_kld:.5f} "
        f"med_KLD={row.median_kld:.5f} top_p={row.same_top_p:.4f}%")


def _free_awq_intermediates(sub: Path) -> None:
    for child in (sub / "model_awq", sub / "model-f16-awq.gguf"):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        elif child.exists():
            child.unlink(missing_ok=True)


# ---- phase 0: setup (fetch, convert, corpora, imatrix, baseline) ----------

def setup() -> None:
    EXP40.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    with phase("[exp-040][setup] extract HF model"):
        step("extract HF", HF_DIR / "config.json",
             lambda: extract.extract_text_lm(source=MODEL_ID, output_dir=HF_DIR))

    with phase("[exp-040][setup] convert -> F16 GGUF"):
        step("convert", F16_GGUF,
             lambda: convert.hf_to_f16_gguf(
                 HF_DIR, F16_GGUF, log=LOGS / "convert.log"))

    with phase("[exp-040][setup] build corpora (cal+eval normal, val=pure MMMU)"):
        def _build_corpora() -> None:
            bc = _load_build_corpora()
            bc.build(
                out_dir=CORPORA,
                model_dir=HF_DIR,
                wiki_test=WIKI_TEST,
                cal_tokens=500_000,
                val_tokens=0,  # MMMU only, no logtrain test slice
                eval_tokens_per_domain=30_000,
                seed=42,
                val_supplement_override=MMMU_VAL,
            )
        step("build_corpora", CORPORA_AUDIT, _build_corpora)

    with phase("[exp-040][setup] imatrix on cal corpus"):
        step("imatrix", IMATRIX,
             lambda: llama_cpp.imatrix(
                 F16_GGUF, CAL_CORPUS, IMATRIX,
                 ctx=IMATRIX_CTX, log=LOGS / "imatrix.log"))

    with phase("[exp-040][setup] F16 baseline KLD on eval corpus"):
        step("baseline", BASELINE_KLD,
             lambda: kld.build_baseline(
                 F16_GGUF, EVAL_CORPUS, BASELINE_KLD,
                 ctx=EVAL_CTX, log=LOGS / "baseline.log"))


# ---- row dispatch ---------------------------------------------------------

def run_imatrix_or_plain(row: Row, *, csv_path: Path, n_params: int) -> None:
    sub = EXP40 / row.label
    sub.mkdir(parents=True, exist_ok=True)
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    qpath = sub / row.gguf_name
    tag = f"[exp-040][{row.label}]"

    with phase(f"{tag} quantize {row.quant} ({row.method})"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 F16_GGUF, qpath, row.quant,
                 imatrix=(IMATRIX if row.use_imatrix else None),
                 log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        method_tag = {"plain": "plain", "imatrix": "imatrix"}[row.method]
        label = f"{MODEL_ID}|{row.quant}|{method_tag} // baseline=FP16"
        _bench(qpath, label, n_params, csv_path, log_dir)


def run_awq(row: Row, *, csv_path: Path, n_params: int) -> None:
    sub = EXP40 / row.label
    sub.mkdir(parents=True, exist_ok=True)
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = sub / "awq.pt"
    model_awq = sub / "model_awq"
    f16_awq = sub / "model-f16-awq.gguf"
    qpath = sub / row.gguf_name

    proxy = row.proxy or awq.proxy_for_quant_type(row.quant)
    tag = f"[exp-040][{row.label}]"

    with phase(f"{tag} AWQ calibrate (proxy={proxy}, mix={row.proxy_mix})"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 HF_DIR, CAL_CORPUS, bundle,
                 proxy=proxy,
                 proxy_mix=row.proxy_mix,
                 proxy_tokens=PROXY_TOKENS,
                 ctx=CTX,
                 device="auto",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 per_tensor_grid_radius=0.15,
                 holdout_text=VAL_CORPUS,
                 cv_strategy="gate",
                 cv_weight=1.0,
             ))

    with phase(f"{tag} AWQ apply"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 HF_DIR, bundle, model_awq,
                 device="auto",
                 dtype="bfloat16",
                 # Qwopus3.5 is Qwen3_5ForConditionalGeneration: its RMSNorm
                 # applies the (1+γ) unit-offset gain, so the fold MUST use
                 # plus_one=True (awq.fold_rmsnorm_gain). plus_one=False (the
                 # gemma-copied setting) left the +1 term unscaled, drifting the
                 # folded F16 forward to apply-rel≈1.0 and collapsing every 2-bit
                 # row (exp-040b confirmed rel→0.40 + recovered top_p on flip).
                 rmsnorm_plus_one=True,
                 # Healthy per-tensor-α drift here is ~0.40 (gemma-like); 0.60
                 # leaves margin yet still rejects the broken-fold rel≈1.0 that
                 # the old 1.20 gate waved through.
                 sanity_max_rel=0.60,
             ))

    with phase(f"{tag} convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=log_dir / "convert.log"))

    with phase(f"{tag} quantize {row.quant}"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, row.quant, imatrix=IMATRIX,
                 log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        mix = row.proxy_mix or "None"
        label = (f"{MODEL_ID}|{row.quant}|awq-cv-gate+imatrix|"
                 f"proxy={proxy},mix={mix},tokens={PROXY_TOKENS},ctx={CTX} "
                 f"// baseline=FP16")
        _bench(qpath, label, n_params, csv_path, log_dir)

    _free_awq_intermediates(sub)


# ---------------------------------------------------------------------------

def main() -> int:
    _check_inputs()
    setup()

    csv_path = EXP40 / "results.csv"
    n_params = bpw_mod.n_params(F16_GGUF)
    log(f"reference n_params (from F16) = {n_params:,.0f}")

    for row in ROWS:
        if row.method == "awq":
            run_awq(row, csv_path=csv_path, n_params=n_params)
        else:
            run_imatrix_or_plain(row, csv_path=csv_path, n_params=n_params)

    log("")
    log("=== exp-040 complete ===")
    log(f"  results: {csv_path}")
    log("  7 vanilla rows expected (Q2_K plain + IQ2_XS/IQ2_M/Q2_K_S × imatrix+AWQ).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
