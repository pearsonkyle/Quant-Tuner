"""exp-044 addendum: IQ3_M + IQ4_XS (plain base imatrix) for the vanilla source.

The original out/exp-044/ vanilla workspace (F16, base imatrix, baseline.kld) was
deleted to free disk, so this script rebuilds those inputs from the cached HF
snapshot and the PRESERVED calibration corpora (now living in the release dir's
calibration_data/), then quantizes + benches IQ3_M and IQ4_XS exactly the way the
shipped IQ2_M / IQ4_XS rows were made: the PLAIN base imatrix collected by
llama-imatrix on the unfolded F16 (NOT hybrid_custom, despite the README prose),
ctx=4096 throughout.

Stages (idempotent via experiments.step(); resumable):
  setup   : copy corpora -> extract vanilla HF -> F16 -> base imatrix -> baseline.kld
  quants  : quantize + KLD-bench IQ3_M and IQ4_XS; append rows to results.csv
  publish : copy the 2 GGUFs into the release dir as gemma-4-31B-it-{QUANT}.gguf

    PYTHONPATH=src .venv/bin/python scripts/exp044_iq3m_iq4xs.py setup
    PYTHONPATH=src .venv/bin/python scripts/exp044_iq3m_iq4xs.py quants
    PYTHONPATH=src .venv/bin/python scripts/exp044_iq3m_iq4xs.py publish
    # or 'all' to chain setup -> quants -> publish.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld
from quant_tuner.bench import runner
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract, llama_cpp
from quant_tuner.quantize import convert, gguf

# ---- identities ------------------------------------------------------------

VANILLA_ID = "google/gemma-4-31B-it"

# ---- isolated exp-044 layout (rebuilds the deleted vanilla workspace) -------

EXP44 = REPO / "out" / "exp-044"
CORPORA = EXP44 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
VAL_CORPUS = CORPORA / "corpus.val.txt"
EVAL_CORPUS = CORPORA / "corpus.eval.txt"
AUDIT = CORPORA / "corpora_audit.json"
BASELINE_KLD = EXP44 / "baseline.kld"
CSV = EXP44 / "results.csv"

VANILLA_DIR = EXP44 / "vanilla"
VANILLA_HF = VANILLA_DIR / "model_extracted"
VANILLA_F16 = VANILLA_DIR / "model-f16.gguf"
VANILLA_IMATRIX = VANILLA_DIR / "imatrix-cal.gguf"

# Renamed release dir (was gemma-4-31B-it-awq-2bit-GGUF). Source of the preserved corpora.
RELEASE = REPO / "uploads" / "pearsonkyle" / "gemma-4-31b-it-imatrix-GGUF"
RELEASE_CAL = RELEASE / "calibration_data"

# ---- knobs (match the exp-044 published config) ----------------------------

IMATRIX_CTX = 4096
EVAL_CTX = 4096

QUANTS: tuple[str, ...] = ("IQ3_M", "IQ4_XS")


def _row_paths(quant: str) -> tuple[Path, Path]:
    """(sub-workspace dir, output GGUF path) for a quant type."""
    label = {"IQ3_M": "v_iq3m_im", "IQ4_XS": "v_iq4xs_im"}[quant]
    sub = EXP44 / label
    return sub, sub / f"gemma-4-31B-it-{quant}-imatrix.gguf"


def _csv_has_qpath(qpath: Path) -> bool:
    if not CSV.exists():
        return False
    needle = str(qpath)
    with CSV.open() as fh:
        return any(needle in line for line in fh)


# ---- setup -----------------------------------------------------------------

def _copy_corpora() -> None:
    CORPORA.mkdir(parents=True, exist_ok=True)
    with phase("[exp-044][setup] stage preserved corpora from release dir"):
        for name in ("corpus.cal.txt", "corpus.val.txt", "corpus.eval.txt",
                     "corpora_audit.json"):
            src = RELEASE_CAL / name
            dst = CORPORA / name
            if dst.exists():
                log(f"  ✓ {name} already present")
                continue
            if not src.exists():
                if name == "corpora_audit.json":
                    log(f"  WARN {name} not in release dir — skipping (non-essential)")
                    continue
                raise FileNotFoundError(f"missing preserved corpus: {src}")
            shutil.copy2(src, dst)
            log(f"  copied {name} ({dst.stat().st_size/1e6:.2f} MB)")


def setup() -> None:
    EXP44.mkdir(parents=True, exist_ok=True)
    VANILLA_DIR.mkdir(parents=True, exist_ok=True)
    logs = VANILLA_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    _copy_corpora()

    with phase("[exp-044][setup] extract vanilla HF (symlink cached snapshot)"):
        if VANILLA_HF.is_symlink():
            VANILLA_HF.unlink()
        step("extract", VANILLA_HF / "config.json",
             lambda: extract.extract_text_lm(source=VANILLA_ID, output_dir=VANILLA_HF))

    with phase("[exp-044][setup] convert vanilla -> F16 GGUF"):
        step("convert", VANILLA_F16,
             lambda: convert.hf_to_f16_gguf(VANILLA_HF, VANILLA_F16,
                                            log=logs / "convert.log"))

    with phase(f"[exp-044][setup] vanilla base imatrix (cal corpus, ctx {IMATRIX_CTX})"):
        step("imatrix", VANILLA_IMATRIX,
             lambda: llama_cpp.imatrix(VANILLA_F16, CAL_CORPUS, VANILLA_IMATRIX,
                                       ctx=IMATRIX_CTX, log=logs / "imatrix.log"))

    with phase(f"[exp-044][setup] vanilla FP16 baseline KLD (eval corpus, ctx {EVAL_CTX})"):
        (EXP44 / "logs").mkdir(parents=True, exist_ok=True)
        step("baseline", BASELINE_KLD,
             lambda: kld.build_baseline(VANILLA_F16, EVAL_CORPUS, BASELINE_KLD,
                                        ctx=EVAL_CTX, log=EXP44 / "logs" / "baseline.log"))

    log("=== exp-044 IQ3_M/IQ4_XS setup complete — inputs ready ===")


# ---- quants ----------------------------------------------------------------

def quants() -> int:
    required = [VANILLA_F16, VANILLA_IMATRIX, EVAL_CORPUS, BASELINE_KLD]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "run `setup` first — missing:\n  " + "\n  ".join(missing))

    n_params = bpw_mod.n_params(VANILLA_F16)

    for quant in QUANTS:
        sub, qpath = _row_paths(quant)
        log_dir = sub / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        tag = f"[exp-044][{sub.name}]"

        with phase(f"{tag} quantize {quant} (imatrix)"):
            step("quantize", qpath,
                 lambda qp=qpath, qt=quant, ld=log_dir: gguf.quantize(
                     VANILLA_F16, qp, qt, imatrix=VANILLA_IMATRIX,
                     log=ld / "quantize.log"))

        with phase(f"{tag} bench"):
            if _csv_has_qpath(qpath):
                log(f"  {qpath.name}: bench row already in CSV — skipping")
                continue
            label = f"{VANILLA_ID}|{quant}|imatrix // baseline=vanilla-FP16"
            row = runner.bench_one(
                qpath, label, reference_n_params=n_params,
                eval_dataset=EVAL_CORPUS, eval_baseline=BASELINE_KLD,
                eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld")
            runner.append_row(CSV, row)
            log(f"  {quant}: bpw={row.bpw:.3f} PPL={row.ppl:.4f} "
                f"med_KLD={row.median_kld:.5f} top_p={row.same_top_p:.4f}%")

    log("")
    log("=== exp-044 IQ3_M/IQ4_XS quants complete ===")
    log(f"  results: {CSV}")
    return 0


# ---- publish ---------------------------------------------------------------

def publish() -> None:
    RELEASE.mkdir(parents=True, exist_ok=True)
    with phase("[exp-044][publish] copy GGUFs into the release dir"):
        for quant in QUANTS:
            _sub, qpath = _row_paths(quant)
            if not qpath.exists():
                log(f"  WARN missing {qpath} — run `quants` first; skipping")
                continue
            # HF tag convention: terminal token = quant -> gemma-4-31B-it-{QUANT}.gguf
            dst = RELEASE / f"gemma-4-31B-it-{quant}.gguf"
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            shutil.copy2(qpath, tmp)
            os.replace(tmp, dst)
            log(f"  published {dst.name} ({dst.stat().st_size/1e9:.1f} GB)")


# ---------------------------------------------------------------------------

def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "setup":
        setup()
    elif stage == "quants":
        return quants()
    elif stage == "publish":
        publish()
    elif stage == "all":
        setup()
        rc = quants()
        if rc != 0:
            return rc
        publish()
    else:
        print(f"unknown stage {stage!r}; use setup|quants|publish|all", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
