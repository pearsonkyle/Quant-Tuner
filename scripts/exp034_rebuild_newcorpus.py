"""Rebuild the gemma-4-31B-it-awq-2bit-GGUF release on the NEW windowed corpus.

The heavy inputs the release builder (exp034_release_v3.py) depends on were
cleaned up after the last upload: no F16 GGUFs, no imatrices, no QAT model. This
orchestrator reconstructs exactly those inputs from the *new* stub+window
calibration corpus, then runs all 11 rows and swaps the GGUFs into the release
directory.

Pipeline:
  1. vanilla: extract google/gemma-4-31B-it (HF-cached) -> F16 GGUF
  2. regenerate EXP20/corpora with the new packer (cal=windowed; val=pure MMMU,
     eval=external — both byte-identical to before, so baseline.kld stays valid)
  3. vanilla imatrix-cal.gguf (ctx 4096) on the new cal corpus; reuse baseline.kld
  4. QAT: download google/gemma-4-31B-it-qat-q4_0-unquantized (~60 GB) -> extract
     -> F16 -> imatrix-cal.gguf on the new cal corpus
  5. run exp034_release_v3.main() (11 rows; QAT IQ2_M intentionally skipped)
  6. copy the 11 result GGUFs + the new corpus into the release dir

Idempotent via experiments.step(): re-runs skip finished stages. Safe to resume.
The existing release GGUFs are left untouched until step 6 overwrites each one
atomically (copy to .tmp then os.replace), so a mid-run failure never leaves a
half-written release file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import kld
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract, llama_cpp
from quant_tuner.quantize import convert

# ---- paths (must match exp034_release_v3.py) -------------------------------

VANILLA_ID = "google/gemma-4-31B-it"
QAT_ID = "google/gemma-4-31B-it-qat-q4_0-unquantized"

EXP20 = REPO / "out" / "exp-020" / VANILLA_ID.replace("/", "__")
EXP22 = REPO / "out" / "exp-022" / QAT_ID.replace("/", "__")
EXP34 = REPO / "out" / "exp-034"

CORPORA = EXP20 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
EVAL_CORPUS = CORPORA / "corpus.eval.txt"
AUDIT = CORPORA / "corpora_audit.json"
BASELINE_KLD = EXP20 / "baseline.kld"
VANILLA_IMATRIX = EXP20 / "imatrix-cal.gguf"

VANILLA_DIR = EXP34 / "vanilla"
VANILLA_HF = VANILLA_DIR / "model_extracted"
VANILLA_F16 = VANILLA_DIR / "model-f16.gguf"

QAT_HF = EXP22 / "model_extracted"
QAT_F16 = EXP22 / "model-f16.gguf"
QAT_IMATRIX = EXP22 / "imatrix-cal.gguf"

WIKI_TEST = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"
MMMU_VAL = REPO / "calibration_supplements" / "mmmu" / "combined.txt"

RELEASE = REPO / "uploads" / "pearsonkyle" / "gemma-4-31B-it-awq-2bit-GGUF"
RELEASE_CAL = RELEASE / "calibration_data"

IMATRIX_CTX = 4096
EVAL_CTX = 4096

# eval corpus md5 from the shipped release; the new packer does not touch the
# eval pack, so this must still match — otherwise baseline.kld reuse is invalid.
EXPECTED_EVAL_MD5 = "9281acdcea259ae0a777e921b11e21b2"


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_module(name: str, filename: str):
    path = REPO / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass type introspection (Py 3.13) can resolve
    # the module's own __dict__; module_from_spec alone does not register it.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_build_corpora():
    return _load_module("build_corpora", "build_corpora.py")


def _load_release_builder():
    return _load_module("exp034_release_v3", "exp034_release_v3.py")


# ---- stages ----------------------------------------------------------------

def setup_vanilla() -> None:
    VANILLA_DIR.mkdir(parents=True, exist_ok=True)
    with phase("[rebuild] extract vanilla HF (cached)"):
        step("extract", VANILLA_HF / "config.json",
             lambda: extract.extract_text_lm(source=VANILLA_ID, output_dir=VANILLA_HF))
    with phase("[rebuild] convert vanilla -> F16 GGUF"):
        step("convert", VANILLA_F16,
             lambda: convert.hf_to_f16_gguf(
                 VANILLA_HF, VANILLA_F16, log=VANILLA_DIR / "convert.log"))


def regen_corpus() -> None:
    CORPORA.mkdir(parents=True, exist_ok=True)
    with phase("[rebuild] regenerate corpora with the new windowed packer"):
        bc = _load_build_corpora()
        # cal=windowed (new defaults); val=pure MMMU (val_tokens=0); eval=external.
        bc.build(
            out_dir=CORPORA,
            model_dir=VANILLA_HF,
            wiki_test=WIKI_TEST,
            cal_tokens=500_000,
            val_tokens=0,
            eval_tokens_per_domain=30_000,
            seed=42,
            val_supplement_override=MMMU_VAL,
        )
    got = _md5(EVAL_CORPUS)
    if got != EXPECTED_EVAL_MD5:
        raise SystemExit(
            f"eval corpus md5 changed ({got} != {EXPECTED_EVAL_MD5}); baseline.kld "
            "is no longer valid — rebuild it before continuing (delete baseline.kld "
            "and add a kld.build_baseline step on VANILLA_F16).")
    log(f"  eval corpus md5 OK ({got}) — baseline.kld reuse is valid")
    # Report the cal windowing gains.
    import json
    w = json.loads(AUDIT.read_text())["calibration"]["pack_audit"].get("windowing")
    if w:
        log(f"  cal corpus: {w['windows_emitted']} windows / {w['sessions_kept']} "
            f"sessions, tool-turn token share {w['tool_turn_token_share']:.0%}")


def vanilla_imatrix() -> None:
    with phase("[rebuild] vanilla imatrix on new cal corpus (ctx 4096)"):
        step("imatrix", VANILLA_IMATRIX,
             lambda: llama_cpp.imatrix(
                 VANILLA_F16, CAL_CORPUS, VANILLA_IMATRIX,
                 ctx=IMATRIX_CTX, log=EXP20 / "logs" / "imatrix-newcorpus.log"))
    if not BASELINE_KLD.exists():
        (EXP20 / "logs").mkdir(parents=True, exist_ok=True)
        with phase("[rebuild] vanilla baseline KLD (eval corpus)"):
            step("baseline", BASELINE_KLD,
                 lambda: kld.build_baseline(
                     VANILLA_F16, EVAL_CORPUS, BASELINE_KLD,
                     ctx=EVAL_CTX, log=EXP20 / "logs" / "baseline-newcorpus.log"))
    else:
        log(f"  reusing existing baseline.kld ({BASELINE_KLD.stat().st_size/1e9:.1f} GB)")


def setup_qat() -> None:
    EXP22.mkdir(parents=True, exist_ok=True)
    (EXP22 / "logs").mkdir(parents=True, exist_ok=True)
    with phase("[rebuild] fetch + extract QAT HF (~60 GB download)"):
        if QAT_HF.is_symlink():
            QAT_HF.unlink()
        step("extract", QAT_HF / "config.json",
             lambda: extract.extract_text_lm(source=QAT_ID, output_dir=QAT_HF))
    with phase("[rebuild] convert QAT -> F16 GGUF"):
        step("convert", QAT_F16,
             lambda: convert.hf_to_f16_gguf(
                 QAT_HF, QAT_F16, log=EXP22 / "logs" / "convert-f16.log"))
    with phase("[rebuild] QAT imatrix on new cal corpus (ctx 4096)"):
        step("imatrix", QAT_IMATRIX,
             lambda: llama_cpp.imatrix(
                 QAT_F16, CAL_CORPUS, QAT_IMATRIX,
                 ctx=IMATRIX_CTX, log=EXP22 / "logs" / "imatrix-newcorpus.log"))


def clean_stale_exp034() -> None:
    """Drop stale bench CSV/log so rows re-bench against the freshly built GGUFs.
    (step() rebuilds the GGUFs themselves since those were already deleted.)"""
    for p in (EXP34 / "results.csv", EXP34 / "run.log"):
        if p.exists():
            p.unlink()
            log(f"  removed stale {p.relative_to(REPO)}")


def publish(rel) -> None:
    """Atomically swap the 11 rebuilt GGUFs + new corpus into the release dir."""
    RELEASE_CAL.mkdir(parents=True, exist_ok=True)
    with phase("[rebuild] publish GGUFs + corpus into the release dir"):
        for row in rel.ROWS:
            src = rel.EXP34 / row.label / row.gguf_name
            if not src.exists():
                log(f"  WARN missing {src} — skipping (row may have failed)")
                continue
            dst = RELEASE / row.gguf_name
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            log(f"  published {row.gguf_name} ({dst.stat().st_size/1e9:.1f} GB)")
        for name in ("corpus.cal.txt", "corpus.val.txt", "corpus.eval.txt",
                     "corpora_audit.json"):
            s = CORPORA / name
            if s.exists():
                shutil.copy2(s, RELEASE_CAL / name)
        log(f"  corpus copied -> {RELEASE_CAL.relative_to(REPO)}")


def main() -> int:
    log("=== exp-034 rebuild on new windowed corpus ===")
    setup_vanilla()
    regen_corpus()
    vanilla_imatrix()
    setup_qat()
    clean_stale_exp034()

    rel = _load_release_builder()
    log("=== inputs ready — handing off to exp034_release_v3 (11 rows) ===")
    rc = rel.main()
    if rc != 0:
        log(f"release builder returned {rc}; not publishing")
        return rc

    publish(rel)
    log("=== rebuild complete ===")
    log(f"  results CSV: {(EXP34 / 'results.csv')}")
    log(f"  release dir: {RELEASE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
