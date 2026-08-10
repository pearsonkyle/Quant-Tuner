"""exp-050 phase 2+3: imatrix + quant ladder for Ornith-1.0-9B (text trunk).

Consumes exp050_setup_ornith.py's outputs (HF dir with the vision tower stripped,
F16 GGUF of the 32-layer text trunk). Builds the corpora + hybrid_custom imatrix,
then the quant lineup.

NOTE: despite `mtp_num_hidden_layers: 1` in the config, Ornith-1.0-9B ships NO
MTP/nextn head weights (verified: 0 nextn tensors in the source safetensors and
in the F16 GGUF, which has only blk.0..31). So — unlike Qwopus3.6 — there is no
draft head to bundle at Q8_0, no tensor pin, and no MTP acceptance eval. These
are plain text quants of a VLM's language trunk.

Lineup (Q2_K plain anchor + 4 hybrid-imatrix rows spanning 2->5 bit):
  1. Q2_K   plain   (no imatrix)      <- no-calibration anchor
  2. IQ2_M  imatrix (hybrid_custom)
  3. IQ3_M  imatrix (hybrid_custom)
  4. IQ4_XS imatrix (hybrid_custom)
  5. Q5_K_S imatrix (hybrid_custom)

Each row is benched twice: on the GENERAL eval corpus (broad English, clean
free-text -> results.general.csv, the §1 published table) and on the TOOLS eval
corpus (in-distribution logtrain holdout -> results.tools.csv; ⚠️ chat markers
tokenize as plain BPE without --parse-special, so those numbers are for
quant-vs-quant comparison only).

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp050_quants_ornith.py
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

MODEL_ID = "deepreinforce-ai/Ornith-1.0-9B"
MODEL_STEM = "Ornith-1.0-9B"

EXP50 = REPO / "out" / "exp-050"
HF_DIR = EXP50 / "model_extracted"
F16_GGUF = EXP50 / "model-f16.gguf"
LOGS = EXP50 / "logs"

CORPORA = EXP50 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
GEN_CORPUS = CORPORA / "corpus.eval.general.txt"
TOOLS_CORPUS = CORPORA / "corpus.eval.tools.txt"
CORPORA_AUDIT = CORPORA / "corpora_audit.json"

BASE_IMATRIX = EXP50 / "imatrix-base.gguf"
HYBRID_IMATRIX = EXP50 / "imatrix-hybrid_custom.gguf"
GEN_BASELINE = EXP50 / "baseline.general.kld"
TOOLS_BASELINE = EXP50 / "baseline.tools.kld"
GEN_CSV = EXP50 / "results.general.csv"
TOOLS_CSV = EXP50 / "results.tools.csv"

WIKI_TEST = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"
MMMU_VAL = REPO / "calibration_supplements" / "mmmu" / "combined.txt"

CTX = 4096
EVAL_CTX = 4096
# Ornith ships NO MTP/nextn head (confirmed: F16 GGUF has only blk.0..31), so
# there is nothing to pin to Q8_0 — plain text-trunk quantization.
NEXTN_PIN: dict[str, str] | None = None


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
    Row("q2k_plain", "Q2_K",   "plain"),    # no-calibration anchor
    Row("iq2m_im",   "IQ2_M",  "imatrix"),
    Row("iq3m_im",   "IQ3_M",  "imatrix"),
    Row("iq4xs_im",  "IQ4_XS", "imatrix"),
    Row("q5ks_im",   "Q5_K_S", "imatrix"),
    Row("q5km_im",   "Q5_K_M", "imatrix"),
)


def _load_build_corpora():
    path = REPO / "scripts" / "build_corpora.py"
    spec = importlib.util.spec_from_file_location("build_corpora", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _csv_has(csv_path: Path, needle: str) -> bool:
    return csv_path.exists() and any(needle in ln for ln in csv_path.open())


def _bench_corpus(row: Row, qpath: Path, sub: Path, n_params: float, *,
                  corpus: Path, baseline: Path, csv_path: Path, tag: str) -> None:
    """Bench one quant against one eval corpus and append a row (idempotent)."""
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with phase(f"{tag} bench {csv_path.stem}"):
        if _csv_has(csv_path, str(qpath)):
            log(f"  {qpath.name}: already in {csv_path.name} — skipping")
            return
        label = (f"{MODEL_ID}|{row.quant}|{row.method} "
                 f"// eval={corpus.stem} // baseline=FP16")
        r = runner.bench_one(
            qpath, label, reference_n_params=n_params,
            eval_dataset=corpus, eval_baseline=baseline,
            eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld",
        )
        runner.append_row(csv_path, r)
        log(f"  {qpath.name}: size={r.size_gib:.2f}GiB bpw={r.bpw:.3f} "
            f"PPL={r.ppl:.4f} medKLD={r.median_kld:.4f} top_p={r.same_top_p:.2f}%")


def main() -> int:
    for p in (HF_DIR / "config.json", F16_GGUF, WIKI_TEST, MMMU_VAL):
        if not p.exists():
            raise FileNotFoundError(f"exp-050 missing input (run setup first): {p}")
    LOGS.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(F16_GGUF)
    log(f"[exp-050] reference n_params (incl. nextn) = {n_params:,.0f}")

    # ---- phase 2: corpora + imatrix + baselines ---------------------------
    with phase("[exp-050] build corpora (Ornith tokenizer)"):
        def _build():
            bc = _load_build_corpora()
            bc.build(
                out_dir=CORPORA, model_dir=HF_DIR, wiki_test=WIKI_TEST,
                cal_tokens=500_000, val_tokens=0, eval_tokens_per_domain=30_000,
                seed=42, val_supplement_override=MMMU_VAL,
            )
        step("build_corpora", CORPORA_AUDIT, _build)

    with phase("[exp-050] base imatrix on cal corpus"):
        step("imatrix-base", BASE_IMATRIX,
             lambda: llama_cpp.imatrix(F16_GGUF, CAL_CORPUS, BASE_IMATRIX,
                                       ctx=CTX, log=LOGS / "imatrix-base.log"))

    with phase("[exp-050] re-weight imatrix -> hybrid_custom"):
        step("imatrix-hybrid", HYBRID_IMATRIX,
             lambda: imatrix_cal.calibrate(
                 variant="hybrid_custom", f16_gguf=F16_GGUF,
                 base_imatrix=BASE_IMATRIX, out_path=HYBRID_IMATRIX))

    with phase("[exp-050] F16 baseline KLD on GENERAL corpus"):
        step("baseline-general", GEN_BASELINE,
             lambda: kld.build_baseline(F16_GGUF, GEN_CORPUS, GEN_BASELINE,
                                        ctx=EVAL_CTX,
                                        log=LOGS / "baseline-general.log"))

    with phase("[exp-050] F16 baseline KLD on TOOLS corpus"):
        step("baseline-tools", TOOLS_BASELINE,
             lambda: kld.build_baseline(F16_GGUF, TOOLS_CORPUS, TOOLS_BASELINE,
                                        ctx=EVAL_CTX,
                                        log=LOGS / "baseline-tools.log"))

    # ---- phase 3: quant rows (nextn pinned Q8_0), bench on both corpora ----
    for row in ROWS:
        sub = EXP50 / row.label
        sub.mkdir(parents=True, exist_ok=True)
        log_dir = sub / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        qpath = sub / row.gguf_name
        imat = None if row.method == "plain" else HYBRID_IMATRIX
        tag = f"[exp-050][{row.label}]"

        with phase(f"{tag} quantize {row.quant} ({row.method})"):
            step("quantize", qpath,
                 lambda q=qpath, qt=row.quant, im=imat, ld=log_dir: gguf.quantize(
                     F16_GGUF, q, qt, imatrix=im,
                     tensor_types=NEXTN_PIN, log=ld / "quantize.log"))

        _bench_corpus(row, qpath, sub, n_params,
                      corpus=GEN_CORPUS, baseline=GEN_BASELINE,
                      csv_path=GEN_CSV, tag=tag)
        _bench_corpus(row, qpath, sub, n_params,
                      corpus=TOOLS_CORPUS, baseline=TOOLS_BASELINE,
                      csv_path=TOOLS_CSV, tag=tag)

    log("")
    log("=== exp-050 phase 2+3 complete ===")
    log(f"  general results: {GEN_CSV}")
    log(f"  tools results:   {TOOLS_CSV}")
    log("  GGUFs:")
    for row in ROWS:
        log(f"    {EXP50 / row.label / row.gguf_name}")
    log("  Next: SWE-rebench agentic eval (run_swebench_eval.py) + README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
