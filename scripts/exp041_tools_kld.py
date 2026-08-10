"""exp-041 follow-up: KLD/PPL of the 4 shipped 2-bit rows on the TOOLS eval corpus.

corpus.eval.tools.txt is the logtrain HOLDOUT slice, windowed with the same
stub+multi-window packer as the calibration corpus (built by build_corpora.py,
2026-06-16). It is the in-distribution tool-call eval that corpus.eval.txt
(external code/math/tools) cannot provide. ⚠️ llama-perplexity has no
--parse-special, so the chat markers tokenize as plain BPE — these numbers are
valid for quant-vs-quant comparison only, not as absolute PPL.

Builds baseline.tools.kld (F16 reference) once, then benches each GGUF against
it. Writes out/exp-041/results.tools.csv.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp041_tools_kld.py
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
TOOLS_CORPUS = EXP41 / "corpora" / "corpus.eval.tools.txt"
TOOLS_BASELINE = EXP41 / "baseline.tools.kld"
CSV = EXP41 / "results.tools.csv"
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
    for p in (F16_GGUF, TOOLS_CORPUS):
        if not p.exists():
            raise FileNotFoundError(p)
    logs = EXP41 / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(F16_GGUF)

    with phase("[exp-041] F16 baseline KLD on TOOLS corpus"):
        step("baseline-tools", TOOLS_BASELINE,
             lambda: kld.build_baseline(F16_GGUF, TOOLS_CORPUS, TOOLS_BASELINE,
                                        ctx=CTX, log=logs / "baseline-tools.log"))

    for quant, method, sub, fname in ROWS:
        qpath = EXP41 / sub / fname
        if not qpath.exists():
            log(f"  SKIP {fname}: missing")
            continue
        with phase(f"[exp-041][{quant}] tools-KLD bench"):
            if _csv_has(CSV, str(qpath)):
                log(f"  {fname}: already in {CSV.name} — skipping")
                continue
            label = f"{MODEL_ID}|{quant}|{method}+nextn@Q8 // eval=TOOLS // baseline=FP16"
            r = runner.bench_one(
                qpath, label, reference_n_params=n_params,
                eval_dataset=TOOLS_CORPUS, eval_baseline=TOOLS_BASELINE,
                eval_ctx=CTX, log_dir=EXP41 / sub / "logs", suite="kld",
            )
            runner.append_row(CSV, r)
            log(f"  {fname}: PPL={r.ppl:.4f} medKLD={r.median_kld:.4f} "
                f"top_p={r.same_top_p:.2f}%")

    log("")
    log(f"=== tools-KLD complete → {CSV} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
