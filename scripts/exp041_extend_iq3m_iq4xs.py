"""exp-041 extension: add IQ3_M + IQ4_XS rows to the Qwopus3.6-27B-Coder release.

The original 4-row 2-bit lineup (exp041_quants_qwopus36.py) shipped, but its
imatrix / baseline.kld / corpora were pruned from out/exp-041 afterward. The F16
GGUF (with the nextn/MTP layer bundled at blk.64) and the extracted HF dir
survive, and the EXACT corpora used for the shipped quants are preserved in the
release package's calibration_data/ (restored into out/exp-041/corpora/ before
running this).

This script reproduces the hybrid_custom imatrix + general baseline.kld from
those preserved corpora, then quantizes two HIGHER-bit rows (IQ3_M, IQ4_XS) with
the same MTP treatment (entire blk.64 nextn layer pinned to Q8_0) and benches
them on the GENERAL eval corpus — matching the "(general)" columns in the
published README table (exp041_general_kld.py).

Idempotent via experiments.step(): the imatrix + baseline are built once and
reused; re-running only fills in missing quant/bench rows.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp041_extend_iq3m_iq4xs.py
"""

from __future__ import annotations

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
F16_GGUF = EXP41 / "model-f16.gguf"
LOGS = EXP41 / "logs"

CORPORA = EXP41 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
GEN_CORPUS = CORPORA / "corpus.eval.general.txt"

BASE_IMATRIX = EXP41 / "imatrix-base.gguf"
HYBRID_IMATRIX = EXP41 / "imatrix-hybrid_custom.gguf"
GEN_BASELINE = EXP41 / "baseline.general.kld"
CSV = EXP41 / "results.general.csv"

CTX = 4096
EVAL_CTX = 4096
# Keep the whole nextn/MTP draft layer (blk.64: attn_*, ffn_*, nextn.*) at Q8_0.
# Substring 'blk.64.' uniquely matches that layer (there is no blk.640+).
NEXTN_PIN = {"blk.64.": "q8_0"}


@dataclass(frozen=True)
class Row:
    label: str
    quant: str

    @property
    def gguf_name(self) -> str:
        return f"{MODEL_STEM}-{self.quant}-imatrix-mtp.gguf"


# Higher-bit additions to the shipped 2-bit lineup. Both imatrix (hybrid_custom).
ROWS = (
    Row("iq3m_im",  "IQ3_M"),
    Row("iq4xs_im", "IQ4_XS"),
)


def _csv_has(path: Path, needle: str) -> bool:
    return path.exists() and any(needle in ln for ln in path.open())


def main() -> int:
    for p in (F16_GGUF, CAL_CORPUS, GEN_CORPUS):
        if not p.exists():
            raise FileNotFoundError(
                f"missing input: {p}\n"
                "  (restore corpora from the release calibration_data/ first)")
    LOGS.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(F16_GGUF)
    log(f"[exp-041 extend] reference n_params (incl. nextn) = {n_params:,.0f}")

    # ---- rebuild imatrix from the preserved calibration corpus -------------
    with phase("[exp-041 extend] base imatrix on cal corpus"):
        step("imatrix-base", BASE_IMATRIX,
             lambda: llama_cpp.imatrix(F16_GGUF, CAL_CORPUS, BASE_IMATRIX,
                                       ctx=CTX, log=LOGS / "imatrix-base.log"))

    with phase("[exp-041 extend] re-weight imatrix -> hybrid_custom"):
        step("imatrix-hybrid", HYBRID_IMATRIX,
             lambda: imatrix_cal.calibrate(
                 variant="hybrid_custom", f16_gguf=F16_GGUF,
                 base_imatrix=BASE_IMATRIX, out_path=HYBRID_IMATRIX))

    with phase("[exp-041 extend] F16 baseline KLD on GENERAL corpus"):
        step("baseline-general", GEN_BASELINE,
             lambda: kld.build_baseline(F16_GGUF, GEN_CORPUS, GEN_BASELINE,
                                        ctx=EVAL_CTX,
                                        log=LOGS / "baseline-general.log"))

    # ---- quantize + bench the two new rows (nextn pinned Q8_0) -------------
    for row in ROWS:
        sub = EXP41 / row.label
        log_dir = sub / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        qpath = sub / row.gguf_name
        tag = f"[exp-041 extend][{row.label}]"

        with phase(f"{tag} quantize {row.quant} (hybrid imatrix, nextn@Q8)"):
            step("quantize", qpath,
                 lambda q=qpath, qt=row.quant, ld=log_dir: gguf.quantize(
                     F16_GGUF, q, qt, imatrix=HYBRID_IMATRIX,
                     tensor_types=NEXTN_PIN, log=ld / "quantize.log"))

        with phase(f"{tag} general-KLD bench"):
            if _csv_has(CSV, str(qpath)):
                log(f"  {qpath.name}: already in {CSV.name} — skipping bench")
                continue
            label = (f"{MODEL_ID}|{row.quant}|imatrix+nextn@Q8 "
                     "// eval=GENERAL // baseline=FP16")
            r = runner.bench_one(
                qpath, label, reference_n_params=n_params,
                eval_dataset=GEN_CORPUS, eval_baseline=GEN_BASELINE,
                eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld",
            )
            runner.append_row(CSV, r)
            log(f"  {qpath.name}: size={r.size_gib:.2f}GiB bpw={r.bpw:.3f} "
                f"PPL={r.ppl:.4f} medKLD={r.median_kld:.4f} "
                f"top_p={r.same_top_p:.2f}%")

    log("")
    log("=== exp-041 extend (IQ3_M + IQ4_XS) complete ===")
    log(f"  results: {CSV}")
    log("  GGUFs:")
    for row in ROWS:
        log(f"    {EXP41 / row.label / row.gguf_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
