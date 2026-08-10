"""exp-045 head-to-head: re-bench Qwopus3.6-27B-Coder on the SAME general eval
corpus + the SAME (current) llama.cpp pin as the tmax-27b quants, so the §1 table
is apples-to-apples.

Qwopus3.6's published README numbers were collected on an older llama.cpp pin. To
compare fairly against tmax-27b (exp-045), we re-bench the four shipped Qwopus GGUFs
against tmax's `corpus.eval.general.txt` (raw external `combined_en_tiny` text — seed
42 deterministic, so byte-identical regardless of which model built it; both models
share the Qwen3.6 tokenizer). No re-quantize: we reuse the already-built Qwopus
upload GGUFs and the Qwopus F16 from exp-041.

Reproduce (after exp045_quants_tmax.py has written the corpora):
    PYTHONPATH=src .venv/bin/python scripts/exp045_compare_qwopus.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.experiments import log, phase

EXP45 = REPO / "out" / "exp-045"
CORPORA = EXP45 / "corpora"
EVAL_GENERAL = CORPORA / "corpus.eval.general.txt"
LOGS = EXP45 / "logs"

QWOPUS_F16 = REPO / "out" / "exp-041" / "model-f16.gguf"
QWOPUS_PKG = REPO / "uploads" / "pearsonkyle" / (
    "Qwopus3.6-27B-Coder-imatrix-2bit-MTP-GGUF"
)
QWOPUS_BASELINE_KLD = EXP45 / "qwopus_baseline.general.kld"
COMPARE_CSV = EXP45 / "qwopus_compare.csv"

EVAL_CTX = 4096
MODEL_ID = "Jackrong/Qwopus3.6-27B-Coder"


@dataclass(frozen=True)
class Row:
    quant: str
    gguf: str
    method: str  # "plain" | "imatrix"


ROWS = (
    Row("Q2_K",   "Qwopus3.6-27B-Coder-Q2_K.gguf",   "plain"),
    Row("IQ2_XS", "Qwopus3.6-27B-Coder-IQ2_XS.gguf", "imatrix"),
    Row("IQ2_M",  "Qwopus3.6-27B-Coder-IQ2_M.gguf",  "imatrix"),
    Row("Q2_K_S", "Qwopus3.6-27B-Coder-Q2_K_S.gguf", "imatrix"),
)


def _csv_has(csv_path: Path, qpath: Path) -> bool:
    if not csv_path.exists():
        return False
    needle = str(qpath)
    with csv_path.open() as fh:
        return any(needle in line for line in fh)


def main() -> int:
    for p in (EVAL_GENERAL, QWOPUS_F16):
        if not p.exists():
            raise FileNotFoundError(
                f"exp-045 compare missing input: {p} "
                "(run exp045_quants_tmax.py first; need its corpus + the exp-041 F16)")
    LOGS.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(QWOPUS_F16)

    with phase("[exp-045] Qwopus FP16 baseline KLD on general eval corpus"):
        if not QWOPUS_BASELINE_KLD.exists():
            kld.build_baseline(QWOPUS_F16, EVAL_GENERAL, QWOPUS_BASELINE_KLD,
                               ctx=EVAL_CTX, log=LOGS / "qwopus_baseline.general.log")
        else:
            log(f"  {QWOPUS_BASELINE_KLD.name}: exists — skipping")

    log(f"[exp-045] Qwopus reference n_params (incl. nextn) = {n_params:,.0f}")

    for row in ROWS:
        qpath = QWOPUS_PKG / row.gguf
        if not qpath.exists():
            raise FileNotFoundError(f"missing Qwopus GGUF: {qpath}")
        log_dir = LOGS / f"qwopus_{row.quant}"
        log_dir.mkdir(parents=True, exist_ok=True)

        with phase(f"[exp-045][qwopus][{row.quant}] bench (general)"):
            if _csv_has(COMPARE_CSV, qpath):
                log(f"  {qpath.name}: already in CSV — skipping bench")
                continue
            label = (f"{MODEL_ID}|{row.quant}|{row.method}+nextn@Q8 "
                     "// baseline=FP16 // eval=general // compare")
            r = runner.bench_one(
                qpath, label, reference_n_params=n_params,
                eval_dataset=EVAL_GENERAL, eval_baseline=QWOPUS_BASELINE_KLD,
                eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld",
            )
            runner.append_row(COMPARE_CSV, r)
            log(f"  {qpath.name}: size={r.size_gib:.2f}GiB bpw={r.bpw:.3f} "
                f"PPL={r.ppl:.4f} medKLD={r.median_kld:.4f} top_p={r.same_top_p:.2f}%")

    log("")
    log("=== exp-045 Qwopus head-to-head bench complete ===")
    log(f"  compare results: {COMPARE_CSV}")
    log("  Use alongside out/exp-045/results.csv for the §1 head-to-head table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
