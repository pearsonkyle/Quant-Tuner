"""exp-041 follow-up: PPL/KLD/top_p of the 4 shipped rows on the GENERAL eval corpus.

corpus.eval.general.txt = ~30k tokens of `combined_en_tiny` (broad English) from
`eaddario/imatrix-calibration` — a clean free-text generalization distribution,
disjoint from calibration. Unlike corpus.eval.tools.txt it is NOT chat-templated,
so llama-perplexity tokenizes it correctly and the numbers are well-formed.

Builds baseline.general.kld (F16 reference) once, then benches each GGUF.
Writes out/exp-041/results.general.csv.

    PYTHONPATH=src .venv/bin/python scripts/exp041_general_kld.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.experiments import log, phase, step

EXP41 = REPO / "out" / "exp-041"
F16_GGUF = EXP41 / "model-f16.gguf"
GEN_CORPUS = EXP41 / "corpora" / "corpus.eval.general.txt"
GEN_BASELINE = EXP41 / "baseline.general.kld"
CSV = EXP41 / "results.general.csv"
CTX = 4096
MODEL_ID = "Jackrong/Qwopus3.6-27B-Coder"

ROWS = (
    ("Q2_K",   "plain",   "q2k_plain", "Qwopus3.6-27B-Coder-Q2_K-plain-mtp.gguf"),
    ("IQ2_XS", "imatrix", "iq2xs_im",  "Qwopus3.6-27B-Coder-IQ2_XS-imatrix-mtp.gguf"),
    ("IQ2_M",  "imatrix", "iq2m_im",   "Qwopus3.6-27B-Coder-IQ2_M-imatrix-mtp.gguf"),
    ("Q2_K_S", "imatrix", "q2ks_im",   "Qwopus3.6-27B-Coder-Q2_K_S-imatrix-mtp.gguf"),
)


def _csv_has(path: Path, needle: str) -> bool:
    return path.exists() and any(needle in ln for ln in path.open())


def main() -> int:
    for p in (F16_GGUF, GEN_CORPUS):
        if not p.exists():
            raise FileNotFoundError(p)
    logs = EXP41 / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(F16_GGUF)

    with phase("[exp-041] F16 baseline KLD on GENERAL corpus"):
        step("baseline-general", GEN_BASELINE,
             lambda: kld.build_baseline(F16_GGUF, GEN_CORPUS, GEN_BASELINE,
                                        ctx=CTX, log=logs / "baseline-general.log"))

    for quant, method, sub, fname in ROWS:
        qpath = EXP41 / sub / fname
        if not qpath.exists():
            log(f"  SKIP {fname}: missing")
            continue
        with phase(f"[exp-041][{quant}] general-KLD bench"):
            if _csv_has(CSV, str(qpath)):
                log(f"  {fname}: already in {CSV.name} — skipping")
                continue
            label = f"{MODEL_ID}|{quant}|{method}+nextn@Q8 // eval=GENERAL // baseline=FP16"
            r = runner.bench_one(
                qpath, label, reference_n_params=n_params,
                eval_dataset=GEN_CORPUS, eval_baseline=GEN_BASELINE,
                eval_ctx=CTX, log_dir=EXP41 / sub / "logs", suite="kld",
            )
            runner.append_row(CSV, r)
            log(f"  {fname}: PPL={r.ppl:.4f} medKLD={r.median_kld:.4f} "
                f"top_p={r.same_top_p:.2f}%")

    log("")
    log(f"=== general-KLD complete → {CSV} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
