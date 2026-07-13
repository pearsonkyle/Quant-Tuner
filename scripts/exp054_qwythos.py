"""exp-054: quantize empero-ai/Qwythos-9B-v2 (IQ2_M / IQ4_XS / Q5_K_M).

Qwythos-9B-v2 is a Qwen3.5-VL 9B (Qwen3_5ForConditionalGeneration) — structurally
identical to Ornith-1.0-9B (32 layers, hidden 4096, hybrid attention full-attn
every 4th, declares mtp_num_hidden_layers=1, thinking model, vision tower). So
this mirrors exp-050/051: strip vision + drop the phantom MTP (keep_mtp=False),
convert to F16, build the agentic-log corpora with Qwythos's tokenizer, collect a
hybrid_custom imatrix, and quantize the ladder. Purpose: compare to Ornith-9B and
test whether the 2-bit agent loop is fixed.

Reproduce:
    PYTHONPATH=src .venv/bin/python scripts/exp054_qwythos.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner
from quant_tuner.calibrate import imatrix as imatrix_cal
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract, llama_cpp
from quant_tuner.quantize import convert, gguf

MODEL_ID = "empero-ai/Qwythos-9B-v2"
MODEL_STEM = "Qwythos-9B-v2"

EXP = REPO / "out" / "exp-054"
HF_DIR = EXP / "model_extracted"
F16_GGUF = EXP / "model-f16.gguf"
LOGS = EXP / "logs"

CORPORA = EXP / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
GEN_CORPUS = CORPORA / "corpus.eval.general.txt"
TOOLS_CORPUS = CORPORA / "corpus.eval.tools.txt"
CORPORA_AUDIT = CORPORA / "corpora_audit.json"

BASE_IMATRIX = EXP / "imatrix-base.gguf"
HYBRID_IMATRIX = EXP / "imatrix-hybrid_custom.gguf"
GEN_BASELINE = EXP / "baseline.general.kld"
TOOLS_BASELINE = EXP / "baseline.tools.kld"
GEN_CSV = EXP / "results.general.csv"
TOOLS_CSV = EXP / "results.tools.csv"

WIKI_TEST = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"
MMMU_VAL = REPO / "calibration_supplements" / "mmmu" / "combined.txt"

CTX = 4096
EVAL_CTX = 4096

# imatrix ladder: the 3 requested quants (2 / 4 / 5 bit)
QUANTS = ("IQ2_M", "IQ4_XS", "Q5_K_M")


def _load_build_corpora():
    spec = importlib.util.spec_from_file_location(
        "build_corpora", REPO / "scripts" / "build_corpora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _csv_has(csv_path: Path, needle: str) -> bool:
    return csv_path.exists() and any(needle in ln for ln in csv_path.open())


def _bench(qpath, n_params, corpus, baseline, csv_path, tag, quant):
    if _csv_has(csv_path, str(qpath)):
        log(f"    {tag}: already benched — skip"); return
    label = f"{MODEL_ID}|{quant}|imatrix // eval={corpus.stem} // baseline=FP16"
    r = runner.bench_one(qpath, label, reference_n_params=n_params,
                         eval_dataset=corpus, eval_baseline=baseline,
                         eval_ctx=EVAL_CTX, log_dir=LOGS, suite="kld")
    runner.append_row(csv_path, r)
    log(f"    {tag}: bpw={r.bpw:.3f} PPL={r.ppl:.4f} medKLD={r.median_kld:.4f} top_p={r.same_top_p:.2f}%")


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    for p in (WIKI_TEST, MMMU_VAL):
        if not p.exists():
            raise FileNotFoundError(f"missing calibration input: {p}")

    # --- extract (strip vision + drop phantom MTP) + convert -----------------
    with phase("[exp-054] extract HF text LM (keep_mtp=False)"):
        step("extract", HF_DIR / "config.json",
             lambda: extract.extract_text_lm(
                 source=MODEL_ID, output_dir=HF_DIR, keep_mtp=False))
    cfg = json.loads((HF_DIR / "config.json").read_text())
    log(f"[exp-054] extracted: layers={cfg.get('num_hidden_layers')} "
        f"mtp={cfg.get('mtp_num_hidden_layers')} arch={cfg.get('architectures')}")
    with phase("[exp-054] convert -> F16 GGUF"):
        step("convert", F16_GGUF,
             lambda: convert.hf_to_f16_gguf(HF_DIR, F16_GGUF, log=LOGS / "convert.log"))

    n_params = bpw_mod.n_params(F16_GGUF)
    log(f"[exp-054] reference n_params = {n_params:,.0f}")

    # --- corpora + imatrix + baselines --------------------------------------
    with phase("[exp-054] build corpora (Qwythos tokenizer)"):
        def _build():
            _load_build_corpora().build(
                out_dir=CORPORA, model_dir=HF_DIR, wiki_test=WIKI_TEST,
                cal_tokens=500_000, val_tokens=0, eval_tokens_per_domain=30_000,
                seed=42, val_supplement_override=MMMU_VAL)
        step("build_corpora", CORPORA_AUDIT, _build)

    with phase("[exp-054] base imatrix"):
        step("imatrix-base", BASE_IMATRIX,
             lambda: llama_cpp.imatrix(F16_GGUF, CAL_CORPUS, BASE_IMATRIX,
                                       ctx=CTX, log=LOGS / "imatrix-base.log"))
    with phase("[exp-054] hybrid_custom imatrix"):
        step("imatrix-hybrid", HYBRID_IMATRIX,
             lambda: imatrix_cal.calibrate(variant="hybrid_custom", f16_gguf=F16_GGUF,
                                           base_imatrix=BASE_IMATRIX, out_path=HYBRID_IMATRIX))
    with phase("[exp-054] FP16 baselines (general + tools)"):
        step("baseline-general", GEN_BASELINE,
             lambda: kld.build_baseline(F16_GGUF, GEN_CORPUS, GEN_BASELINE,
                                        ctx=EVAL_CTX, log=LOGS / "baseline-general.log"))
        step("baseline-tools", TOOLS_BASELINE,
             lambda: kld.build_baseline(F16_GGUF, TOOLS_CORPUS, TOOLS_BASELINE,
                                        ctx=EVAL_CTX, log=LOGS / "baseline-tools.log"))

    # --- quantize + bench ----------------------------------------------------
    for quant in QUANTS:
        sub = EXP / quant.lower()
        sub.mkdir(parents=True, exist_ok=True)
        qpath = sub / f"{MODEL_STEM}-{quant}.gguf"
        with phase(f"[exp-054] quantize {quant}"):
            step("quantize", qpath,
                 lambda q=qpath, t=quant: gguf.quantize(
                     F16_GGUF, q, t, imatrix=HYBRID_IMATRIX, log=sub / "quantize.log"))
        with phase(f"[exp-054] bench {quant}"):
            _bench(qpath, n_params, GEN_CORPUS, GEN_BASELINE, GEN_CSV, "general", quant)
            _bench(qpath, n_params, TOOLS_CORPUS, TOOLS_BASELINE, TOOLS_CSV, "tools", quant)

    log("")
    log("=== exp-054 complete ===")
    for quant in QUANTS:
        log(f"  {EXP / quant.lower() / f'{MODEL_STEM}-{quant}.gguf'}")
    log("  Next: SWE-rebench on the IQ2_M (1 rep, 10 instances) — loop test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
