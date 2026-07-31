"""Experiment 034: full release re-quant at proxy=1024 ctx=4096 mix=None.

Reproduces every row of uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/README.md
except QAT IQ2_M (broken: PPL 2.2e10 in exp-033 regardless of proxy choice).

Settings learned from exp-031:
  * AWQ proxy_tokens = 1024 (medKLD 1.151 vs proxy=512's 1.192 on QAT IQ2_XS)
  * AWQ ctx          = 4096
  * AWQ proxy_mix    = None (mix=ON regressed in exp-030, predicate-correct)
  * IQ2_M AWQ        = pin proxy=q2k_b16 (shipped recipe workaround)
  * cv_strategy      = gate, cv_weight = 1.0
  * imatrix          = exp-020 (vanilla) / exp-022 (QAT), both ctx=4096

11 rows:
  1.  vanilla Q2_K     plain (no imatrix, no AWQ)
  2.  vanilla IQ2_XS   imatrix only
  3.  vanilla IQ2_XS   AWQ cv-gate + imatrix
  4.  vanilla IQ2_M    imatrix only
  5.  vanilla IQ2_M    AWQ cv-gate + imatrix (proxy=q2k_b16)
  6.  vanilla Q2_K_S   imatrix only
  7.  vanilla Q2_K_S   AWQ cv-gate + imatrix
  8.  QAT     IQ2_XS   imatrix only
  9.  QAT     IQ2_XS   AWQ cv-gate + imatrix
  10. QAT     Q2_K_S   imatrix only
  11. QAT     Q2_K_S   AWQ cv-gate + imatrix

Phase 0 fetches the vanilla HF model (~60 GB) and converts to F16 GGUF.
Each AWQ row auto-deletes its ~115 GB intermediates after bench lands.
Bench rows dedup against the CSV so re-runs pick up where they died.

Wall time: ~15-18 h on Metal.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate import awq
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract
from quant_tuner.quantize import convert, gguf

# ---- inputs ----------------------------------------------------------------

VANILLA_ID = "google/gemma-4-31B-it"
VANILLA_SLUG = VANILLA_ID.replace("/", "__")
QAT_ID = "google/gemma-4-31B-it-qat-q4_0-unquantized"
QAT_SLUG = QAT_ID.replace("/", "__")

EXP20 = REPO / "out" / "exp-020" / VANILLA_SLUG
EXP22 = REPO / "out" / "exp-022" / QAT_SLUG
EXP34 = REPO / "out" / "exp-034"

CORPORA = EXP20 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
VAL_CORPUS = CORPORA / "corpus.val.txt"
EVAL_CORPUS = CORPORA / "corpus.eval.txt"
BASELINE_KLD = EXP20 / "baseline.kld"

VANILLA_IMATRIX = EXP20 / "imatrix-cal.gguf"
QAT_HF = EXP22 / "model_extracted"
QAT_F16 = EXP22 / "model-f16.gguf"
QAT_IMATRIX = EXP22 / "imatrix-cal.gguf"

VANILLA_DIR = EXP34 / "vanilla"
VANILLA_HF = VANILLA_DIR / "model_extracted"
VANILLA_F16 = VANILLA_DIR / "model-f16.gguf"
VANILLA_LOGS = VANILLA_DIR / "logs"

PROXY_TOKENS = 1024
CTX = 4096
EVAL_CTX = 4096


# ---- table -----------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    label: str            # short tag for paths
    source: str           # "vanilla" or "qat"
    quant: str            # IQ2_XS, IQ2_M, Q2_K_S, Q2_K
    method: str           # "plain", "imatrix", "awq"
    proxy: str | None = None  # AWQ proxy override; None = auto from quant_type

    @property
    def use_imatrix(self) -> bool:
        return self.method in ("imatrix", "awq")

    @property
    def gguf_name(self) -> str:
        if self.source == "vanilla":
            stem = f"gemma-4-31B-it-{self.quant}"
        else:
            stem = f"gemma-4-31B-it-qat-{self.quant}"
        if self.method == "plain":
            return f"{stem}-plain.gguf"
        if self.method == "imatrix":
            return f"{stem}-imatrix.gguf"
        return f"{stem}-awq-cv-gate.gguf"


ROWS: tuple[Row, ...] = (
    Row("v_q2k_plain",   "vanilla", "Q2_K",   "plain"),
    Row("v_iq2xs_im",    "vanilla", "IQ2_XS", "imatrix"),
    Row("v_iq2xs_awq",   "vanilla", "IQ2_XS", "awq"),
    Row("v_iq2m_im",     "vanilla", "IQ2_M",  "imatrix"),
    Row("v_iq2m_awq",    "vanilla", "IQ2_M",  "awq", proxy="q2k_b16"),
    Row("v_q2ks_im",     "vanilla", "Q2_K_S", "imatrix"),
    Row("v_q2ks_awq",    "vanilla", "Q2_K_S", "awq"),
    Row("q_iq2xs_im",    "qat",     "IQ2_XS", "imatrix"),
    Row("q_iq2xs_awq",   "qat",     "IQ2_XS", "awq"),
    Row("q_q2ks_im",     "qat",     "Q2_K_S", "imatrix"),
    Row("q_q2ks_awq",    "qat",     "Q2_K_S", "awq"),
)


# ---------------------------------------------------------------------------

def _check_inputs() -> None:
    required = [
        QAT_HF / "config.json", QAT_F16, QAT_IMATRIX, VANILLA_IMATRIX,
        CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("exp-034 missing inputs:\n  " + "\n  ".join(missing))


def _csv_has_qpath(csv_path: Path, qpath: Path) -> bool:
    if not csv_path.exists():
        return False
    needle = str(qpath)
    with csv_path.open() as fh:
        for line in fh:
            if needle in line:
                return True
    return False


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


# ---- phase 0: vanilla setup -----------------------------------------------

def setup_vanilla() -> None:
    VANILLA_DIR.mkdir(parents=True, exist_ok=True)
    VANILLA_LOGS.mkdir(parents=True, exist_ok=True)
    with phase("[exp-034][setup] extract vanilla HF model"):
        step("extract HF", VANILLA_HF / "config.json",
             lambda: extract.extract_text_lm(
                 source=VANILLA_ID, output_dir=VANILLA_HF))
    with phase("[exp-034][setup] convert vanilla -> F16 GGUF"):
        step("convert", VANILLA_F16,
             lambda: convert.hf_to_f16_gguf(
                 VANILLA_HF, VANILLA_F16, log=VANILLA_LOGS / "convert.log"))


# ---- row dispatch ---------------------------------------------------------

def _src_paths(source: str) -> tuple[Path, Path, Path]:
    """Return (HF dir, F16 GGUF, imatrix) for a source."""
    if source == "vanilla":
        return VANILLA_HF, VANILLA_F16, VANILLA_IMATRIX
    return QAT_HF, QAT_F16, QAT_IMATRIX


def run_imatrix_or_plain(row: Row, *, csv_path: Path) -> None:
    src_hf, src_f16, imatrix = _src_paths(row.source)
    sub = EXP34 / row.label
    sub.mkdir(parents=True, exist_ok=True)
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    qpath = sub / row.gguf_name
    n_params = bpw_mod.n_params(src_f16)
    tag = f"[exp-034][{row.label}]"

    with phase(f"{tag} quantize {row.quant} ({row.method})"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 src_f16, qpath, row.quant,
                 imatrix=(imatrix if row.use_imatrix else None),
                 log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        suffix = "qat-" if row.source == "qat" else ""
        method_tag = {"plain": "plain", "imatrix": "imatrix",
                      "awq": "awq-cv-gate+imatrix"}[row.method]
        label = (f"{VANILLA_ID if row.source == 'vanilla' else QAT_ID}"
                 f"|{row.quant}|{method_tag} // baseline=vanilla-FP16")
        _bench(qpath, label, n_params, csv_path, log_dir)


def run_awq(row: Row, *, csv_path: Path) -> None:
    src_hf, src_f16, imatrix = _src_paths(row.source)
    sub = EXP34 / row.label
    sub.mkdir(parents=True, exist_ok=True)
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = sub / "awq.pt"
    model_awq = sub / "model_awq"
    f16_awq = sub / "model-f16-awq.gguf"
    qpath = sub / row.gguf_name
    n_params = bpw_mod.n_params(src_f16)

    proxy = row.proxy or awq.proxy_for_quant_type(row.quant)
    tag = f"[exp-034][{row.label}]"

    with phase(f"{tag} AWQ calibrate (proxy={proxy})"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 src_hf, CAL_CORPUS, bundle,
                 proxy=proxy,
                 proxy_mix=None,
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
                 src_hf, bundle, model_awq,
                 device="auto",
                 dtype="bfloat16",
                 rmsnorm_plus_one=False,
                 sanity_max_rel=1.20,
             ))

    with phase(f"{tag} convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(
                 model_awq, f16_awq, log=log_dir / "convert.log"))

    with phase(f"{tag} quantize {row.quant}"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 f16_awq, qpath, row.quant, imatrix=imatrix,
                 log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        label = (f"{VANILLA_ID if row.source == 'vanilla' else QAT_ID}"
                 f"|{row.quant}|awq-cv-gate+imatrix|"
                 f"proxy={proxy},mix=None,tokens={PROXY_TOKENS},ctx={CTX} "
                 f"// baseline=vanilla-FP16")
        _bench(qpath, label, n_params, csv_path, log_dir)

    _free_awq_intermediates(sub)


# ---------------------------------------------------------------------------

def main() -> int:
    _check_inputs()
    EXP34.mkdir(parents=True, exist_ok=True)
    csv_path = EXP34 / "results.csv"

    # Phase 0: fetch + convert vanilla if any vanilla row needs it.
    if any(r.source == "vanilla" for r in ROWS):
        setup_vanilla()

    for row in ROWS:
        if row.method == "awq":
            run_awq(row, csv_path=csv_path)
        else:
            run_imatrix_or_plain(row, csv_path=csv_path)

    log("")
    log("=== exp-034 complete ===")
    log(f"  results: {csv_path}")
    log("  11 rows expected (vanilla×7 + QAT×4; QAT IQ2_M skipped).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
