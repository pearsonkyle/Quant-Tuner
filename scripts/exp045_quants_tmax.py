"""exp-045 phase 2+3: imatrix + 2-bit quants for allenai/tmax-27b (text-only).

Consumes exp045_setup_tmax.py's outputs (extracted HF dir + F16 GGUF). Builds the
corpora + hybrid_custom imatrix, then the 4-row 2-bit lineup.

NOTE: tmax-27b ships NO MTP head. The source checkpoint contains only the 64-layer
text trunk (`model.language_model.*` + lm_head) — the `mtp_num_hidden_layers=1` in
config.json is a vestigial flag inherited from Qwen3.6 base; the actual nextn/MTP
weights were never released. So there is no blk.64 to pin and no speculative-decoding
draft head. Unlike the Qwopus3.6 release, this is a plain 2-bit imatrix release.

Lineup:
  1. Q2_K   plain  (no imatrix)
  2. IQ2_XS imatrix (hybrid_custom)
  3. IQ2_M  imatrix (hybrid_custom)
  4. Q2_K_S imatrix (hybrid_custom)

Bench is on corpus.eval.general.txt (the broad-English free-text eval — the columns
the shipped README §1 table renders). The corpora builder also writes
corpus.eval.tools.txt (in-distribution); its baseline is built here too.

After this, run the agentic head-to-head (run_swebench_eval.py) before publishing.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp045_quants_tmax.py
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

MODEL_ID = "allenai/tmax-27b"
MODEL_STEM = "tmax-27b"

EXP45 = REPO / "out" / "exp-045"
HF_DIR = EXP45 / "model_extracted"
F16_GGUF = EXP45 / "model-f16.gguf"
LOGS = EXP45 / "logs"

CORPORA = EXP45 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
EVAL_GENERAL = CORPORA / "corpus.eval.general.txt"
EVAL_TOOLS = CORPORA / "corpus.eval.tools.txt"
CORPORA_AUDIT = CORPORA / "corpora_audit.json"

BASE_IMATRIX = EXP45 / "imatrix-base.gguf"
HYBRID_IMATRIX = EXP45 / "imatrix-hybrid_custom.gguf"
BASELINE_GENERAL_KLD = EXP45 / "baseline.general.kld"
BASELINE_TOOLS_KLD = EXP45 / "baseline.tools.kld"

WIKI_TEST = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"
MMMU_VAL = REPO / "calibration_supplements" / "mmmu" / "combined.txt"

CTX = 4096
EVAL_CTX = 4096


@dataclass(frozen=True)
class Row:
    label: str
    quant: str
    method: str  # "plain" | "imatrix"

    @property
    def gguf_name(self) -> str:
        suffix = "plain" if self.method == "plain" else "imatrix"
        return f"{MODEL_STEM}-{self.quant}-{suffix}.gguf"


ROWS = (
    Row("q2k_plain", "Q2_K",   "plain"),
    Row("iq2xs_im",  "IQ2_XS", "imatrix"),
    Row("iq2m_im",   "IQ2_M",  "imatrix"),
    Row("q2ks_im",   "Q2_K_S", "imatrix"),
    # Higher-bit additions (reuse the same hybrid imatrix + F16 + baselines).
    Row("iq3m_im",   "IQ3_M",  "imatrix"),
    Row("iq4xs_im",  "IQ4_XS", "imatrix"),
    Row("q5km_im",   "Q5_K_M", "imatrix"),
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
            raise FileNotFoundError(f"exp-045 missing input (run setup first): {p}")
    LOGS.mkdir(parents=True, exist_ok=True)
    csv_path = EXP45 / "results.csv"
    n_params = bpw_mod.n_params(F16_GGUF)

    # ---- phase 2: corpora + imatrix + baseline ----------------------------
    with phase("[exp-045] build corpora (tmax tokenizer)"):
        def _build():
            bc = _load_build_corpora()
            bc.build(
                out_dir=CORPORA, model_dir=HF_DIR, wiki_test=WIKI_TEST,
                cal_tokens=500_000, val_tokens=0, eval_tokens_per_domain=30_000,
                seed=42, val_supplement_override=MMMU_VAL,
            )
        step("build_corpora", CORPORA_AUDIT, _build)

    with phase("[exp-045] base imatrix on cal corpus"):
        step("imatrix-base", BASE_IMATRIX,
             lambda: llama_cpp.imatrix(F16_GGUF, CAL_CORPUS, BASE_IMATRIX,
                                       ctx=CTX, log=LOGS / "imatrix-base.log"))

    with phase("[exp-045] re-weight imatrix -> hybrid_custom"):
        step("imatrix-hybrid", HYBRID_IMATRIX,
             lambda: imatrix_cal.calibrate(
                 variant="hybrid_custom", f16_gguf=F16_GGUF,
                 base_imatrix=BASE_IMATRIX, out_path=HYBRID_IMATRIX))

    with phase("[exp-045] F16 baseline KLD on general eval corpus"):
        step("baseline-general", BASELINE_GENERAL_KLD,
             lambda: kld.build_baseline(F16_GGUF, EVAL_GENERAL, BASELINE_GENERAL_KLD,
                                        ctx=EVAL_CTX, log=LOGS / "baseline.general.log"))

    with phase("[exp-045] F16 baseline KLD on tools eval corpus"):
        step("baseline-tools", BASELINE_TOOLS_KLD,
             lambda: kld.build_baseline(F16_GGUF, EVAL_TOOLS, BASELINE_TOOLS_KLD,
                                        ctx=EVAL_CTX, log=LOGS / "baseline.tools.log"))

    log(f"[exp-045] reference n_params = {n_params:,.0f}")

    # ---- phase 3: 4 quant rows --------------------------------------------
    for row in ROWS:
        sub = EXP45 / row.label
        sub.mkdir(parents=True, exist_ok=True)
        log_dir = sub / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        qpath = sub / row.gguf_name
        imat = None if row.method == "plain" else HYBRID_IMATRIX
        tag = f"[exp-045][{row.label}]"

        with phase(f"{tag} quantize {row.quant} ({row.method})"):
            step("quantize", qpath,
                 lambda q=qpath, qt=row.quant, im=imat, ld=log_dir: gguf.quantize(
                     F16_GGUF, q, qt, imatrix=im, log=ld / "quantize.log"))

        with phase(f"{tag} bench (general)"):
            if _csv_has(csv_path, qpath):
                log(f"  {qpath.name}: already in CSV — skipping bench")
                continue
            label = f"{MODEL_ID}|{row.quant}|{row.method} // baseline=FP16 // eval=general"
            r = runner.bench_one(
                qpath, label, reference_n_params=n_params,
                eval_dataset=EVAL_GENERAL, eval_baseline=BASELINE_GENERAL_KLD,
                eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld",
            )
            runner.append_row(csv_path, r)
            log(f"  {qpath.name}: size={r.size_gib:.2f}GiB bpw={r.bpw:.3f} "
                f"PPL={r.ppl:.4f} medKLD={r.median_kld:.4f} top_p={r.same_top_p:.2f}%")

    log("")
    log("=== exp-045 phase 2+3 complete ===")
    log(f"  results: {csv_path}")
    log("  Next: agentic head-to-head (run_swebench_eval.py: tmax IQ2_M vs Qwopus3.6")
    log("  IQ2_M, both without --spec-type — tmax has no MTP head).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
