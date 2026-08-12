"""exp-041 phase 2+3: imatrix + 2-bit quants for Qwopus3.6-27B-Coder, MTP-kept.

Consumes exp041_setup_qwopus36.py's outputs (HF dir with vision stripped + MTP
kept, F16 GGUF with the nextn draft layer at blk.64). Builds the corpora +
hybrid_custom imatrix, then the 4-row 2-bit lineup, pinning the ENTIRE nextn
draft layer (all of blk.64 — attn/ffn/eh_proj/norms) to Q8_0 so the MTP head
stays near-lossless and actually drafts well, while the trunk goes to 2-bit.

Lineup (mirrors the shipped imatrix-only release shape):
  1. Q2_K   plain  (no imatrix)            + nextn@Q8
  2. IQ2_XS imatrix (hybrid_custom)        + nextn@Q8
  3. IQ2_M  imatrix (hybrid_custom)        + nextn@Q8
  4. Q2_K_S imatrix (hybrid_custom)        + nextn@Q8

After this, bench MTP acceptance/speed separately (bench_mtp_speed.py) to
confirm the draft head gives a real speedup before publishing.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp041_quants_qwopus36.py
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.calibrate import imatrix as imatrix_cal
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import llama_cpp
from quant_tuner.quantize import gguf

MODEL_ID = "Jackrong/Qwopus3.6-27B-Coder"
MODEL_STEM = "Qwopus3.6-27B-Coder"

EXP41 = REPO / "out" / "exp-041"
HF_DIR = EXP41 / "model_extracted"
F16_GGUF = EXP41 / "model-f16.gguf"
LOGS = EXP41 / "logs"

CORPORA = EXP41 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
EVAL_CORPUS = CORPORA / "corpus.eval.txt"
CORPORA_AUDIT = CORPORA / "corpora_audit.json"

BASE_IMATRIX = EXP41 / "imatrix-base.gguf"
HYBRID_IMATRIX = EXP41 / "imatrix-hybrid_custom.gguf"
BASELINE_KLD = EXP41 / "baseline.kld"

WIKI_TEST = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"
MMMU_VAL = REPO / "calibration_supplements" / "mmmu" / "combined.txt"

CTX = 4096
EVAL_CTX = 4096
# Keep the whole nextn/MTP draft layer (blk.64: attn_*, ffn_*, nextn.*) at Q8_0.
# Substring 'blk.64.' uniquely matches that layer (there is no blk.640+).
NEXTN_PIN = {"blk.64.": "q8_0"}


@dataclass(frozen=True)
class Row:
    label: str
    quant: str
    method: str  # "plain" | "imatrix"

    @property
    def gguf_name(self) -> str:
        suffix = "plain" if self.method == "plain" else "imatrix"
        return f"{MODEL_STEM}-{self.quant}-{suffix}-mtp.gguf"


ROWS = (
    Row("q2k_plain", "Q2_K",   "plain"),
    Row("iq2xs_im",  "IQ2_XS", "imatrix"),
    Row("iq2m_im",   "IQ2_M",  "imatrix"),
    Row("q2ks_im",   "Q2_K_S", "imatrix"),
)


def _load_build_corpora():
    path = REPO / "scripts" / "build_corpora.py"
    spec = importlib.util.spec_from_file_location("build_corpora", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _csv_has(csv_path: Path, qpath: Path) -> bool:
    if not csv_path.exists():
        return False
    needle = str(qpath)
    with csv_path.open() as fh:
        return any(needle in line for line in fh)


def main() -> int:
    for p in (HF_DIR / "config.json", F16_GGUF, WIKI_TEST, MMMU_VAL):
        if not p.exists():
            raise FileNotFoundError(f"exp-041 missing input (run setup first): {p}")
    LOGS.mkdir(parents=True, exist_ok=True)
    csv_path = EXP41 / "results.csv"
    n_params = bpw_mod.n_params(F16_GGUF)

    # ---- phase 2: corpora + imatrix + baseline ----------------------------
    with phase("[exp-041] build corpora (new tokenizer)"):
        def _build():
            bc = _load_build_corpora()
            bc.build(
                out_dir=CORPORA, model_dir=HF_DIR, wiki_test=WIKI_TEST,
                cal_tokens=500_000, val_tokens=0, eval_tokens_per_domain=30_000,
                seed=42, val_supplement_override=MMMU_VAL,
            )
        step("build_corpora", CORPORA_AUDIT, _build)

    with phase("[exp-041] base imatrix on cal corpus"):
        step("imatrix-base", BASE_IMATRIX,
             lambda: llama_cpp.imatrix(F16_GGUF, CAL_CORPUS, BASE_IMATRIX,
                                       ctx=CTX, log=LOGS / "imatrix-base.log"))

    with phase("[exp-041] re-weight imatrix -> hybrid_custom"):
        step("imatrix-hybrid", HYBRID_IMATRIX,
             lambda: imatrix_cal.calibrate(
                 variant="hybrid_custom", f16_gguf=F16_GGUF,
                 base_imatrix=BASE_IMATRIX, out_path=HYBRID_IMATRIX))

    with phase("[exp-041] F16 baseline KLD on eval corpus"):
        step("baseline", BASELINE_KLD,
             lambda: kld.build_baseline(F16_GGUF, EVAL_CORPUS, BASELINE_KLD,
                                        ctx=EVAL_CTX, log=LOGS / "baseline.log"))

    log(f"[exp-041] reference n_params (incl. nextn) = {n_params:,.0f}")

    # ---- phase 3: 4 quant rows (nextn pinned Q8_0) ------------------------
    for row in ROWS:
        sub = EXP41 / row.label
        sub.mkdir(parents=True, exist_ok=True)
        log_dir = sub / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        qpath = sub / row.gguf_name
        imat = None if row.method == "plain" else HYBRID_IMATRIX
        tag = f"[exp-041][{row.label}]"

        with phase(f"{tag} quantize {row.quant} ({row.method}, nextn@Q8)"):
            step("quantize", qpath,
                 lambda q=qpath, qt=row.quant, im=imat, ld=log_dir: gguf.quantize(
                     F16_GGUF, q, qt, imatrix=im,
                     tensor_types=NEXTN_PIN, log=ld / "quantize.log"))

        with phase(f"{tag} bench"):
            if _csv_has(csv_path, qpath):
                log(f"  {qpath.name}: already in CSV — skipping bench")
                continue
            label = f"{MODEL_ID}|{row.quant}|{row.method}+nextn@Q8 // baseline=FP16"
            r = runner.bench_one(
                qpath, label, reference_n_params=n_params,
                eval_dataset=EVAL_CORPUS, eval_baseline=BASELINE_KLD,
                eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld",
            )
            runner.append_row(csv_path, r)
            log(f"  {qpath.name}: size={r.size_gib:.2f}GiB bpw={r.bpw:.3f} "
                f"PPL={r.ppl:.4f} medKLD={r.median_kld:.4f} top_p={r.same_top_p:.2f}%")

    log("")
    log("=== exp-041 phase 2+3 complete ===")
    log(f"  results: {csv_path}")
    log("  Next: bench MTP acceptance/speed on these GGUFs (--spec-type draft-mtp).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
