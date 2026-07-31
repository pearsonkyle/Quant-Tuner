"""exp-044: reproduce the gemma-4-31B-it AWQ 2-bit release on UPDATED llama.cpp
+ the Qwopus3.6-style windowed calibration corpus.

This is a clean-room re-run of the 11-row release (cf. exp034_release_v3.py and
exp034_rebuild_newcorpus.py), isolated under out/exp-044 so the original exp-020/
exp-034 records stay frozen. Three deliberate differences from exp-034:

  1. Calibration data: the windowed stub+multi-window packer (build_corpora.py
     defaults), with the VALIDATION slice taken from the logtrain TEST split
     (Qwopus3.6 recipe) — NOT the MMMU supplement the original gemma release used.
  2. AWQ proxies: q2k_super for Q2_K_S; q2k_b16 + iq3_s mix for IQ2_M; auto iq2_xs
     for IQ2_XS. (Confirm/override these from the Stage C0 sweep — see PROXY/MIX
     constants below.)
  3. Toolchain: the rebuilt llama.cpp + regenerated IQ2 grids. Because FP16
     reference logits can shift across a llama bump, baseline.kld is rebuilt here
     (not reused from a prior release).

Faithful to the gemma scripts (NOT the Qwopus exp-042 flow): the final
llama-quantize uses the PLAIN base imatrix collected on the unfolded F16 — the
gemma release never folded the imatrix or re-weighted it to hybrid_custom, despite
the README prose. rmsnorm_plus_one=False (Gemma is not a (1+gamma) norm).

Stages (idempotent via experiments.step(); resumable):
  setup   : extract vanilla+QAT HF -> F16 -> base imatrix; build corpus; baseline.kld
  quants  : 11 rows (vanilla x7 + QAT x4; QAT IQ2_M intentionally skipped)
  publish : copy the 11 GGUFs + corpus into the release dir

    PYTHONPATH=src .venv/bin/python scripts/exp044_release_reproduce.py setup
    PYTHONPATH=src .venv/bin/python scripts/exp044_release_reproduce.py quants
    PYTHONPATH=src .venv/bin/python scripts/exp044_release_reproduce.py publish
    # or 'all' to chain setup -> quants -> publish.

Wall time: vanilla AWQ leg runs on CPU (won't fit MPS); ~15-18 h for 11 rows on
Metal. Each AWQ row auto-frees its ~115 GB intermediates after bench.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld
from quant_tuner.bench import runner
from quant_tuner.calibrate import awq
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract, llama_cpp
from quant_tuner.quantize import convert, gguf

# ---- identities ------------------------------------------------------------

VANILLA_ID = "google/gemma-4-31B-it"
QAT_ID = "google/gemma-4-31B-it-qat-q4_0-unquantized"

# ---- isolated exp-044 layout ----------------------------------------------

EXP44 = REPO / "out" / "exp-044"
CORPORA = EXP44 / "corpora"
CAL_CORPUS = CORPORA / "corpus.cal.txt"
VAL_CORPUS = CORPORA / "corpus.val.txt"
EVAL_CORPUS = CORPORA / "corpus.eval.txt"
AUDIT = CORPORA / "corpora_audit.json"
BASELINE_KLD = EXP44 / "baseline.kld"

VANILLA_DIR = EXP44 / "vanilla"
VANILLA_HF = VANILLA_DIR / "model_extracted"
VANILLA_F16 = VANILLA_DIR / "model-f16.gguf"
VANILLA_IMATRIX = VANILLA_DIR / "imatrix-cal.gguf"

QAT_DIR = EXP44 / "qat"
# Reuse the existing exp-022 QAT extracted-HF + F16 (deterministic, llama-version-
# independent weight copy) to skip a ~60 GB re-extract and a slow 61 GB reconvert;
# only the imatrix is re-collected on the new corpus (into exp-044).
_EXP22 = REPO / "out" / "exp-022" / QAT_ID.replace("/", "__")
QAT_HF = _EXP22 / "model_extracted"
QAT_F16 = _EXP22 / "model-f16.gguf"
QAT_IMATRIX = QAT_DIR / "imatrix-cal.gguf"

WIKI_TEST = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"

RELEASE = REPO / "uploads" / "pearsonkyle" / "gemma-4-31B-it-awq-2bit-GGUF"
RELEASE_CAL = RELEASE / "calibration_data"

# ---- knobs (match exp-034 winner config from exp-031) ----------------------

IMATRIX_CTX = 4096
CTX = 4096
EVAL_CTX = 4096
PROXY_TOKENS = 1024
PER_TENSOR_GRID_RADIUS = 0.15
SANITY_MAX_REL = 1.20
RMSNORM_PLUS_ONE = False  # Gemma — see reference_awq_arch_settings memory

# ---- AWQ proxy selection (UPDATE from the Stage C0 sweep if a rival wins) ---
# Defaults reproduce the README text. proxy=None -> awq.proxy_for_quant_type().
PROXY: dict[str, str | None] = {
    "IQ2_XS": None,        # auto iq2_xs codebook (sweep winner: medKLD 1.790 vs 2.220)
    "IQ2_M": "q2k_b16",    # sweep winner: medKLD 1.602 w/ mix vs 1.699 no-mix
    "Q2_K_S": "q2k_b16",   # sweep winner: medKLD 1.719 vs 2.027 (q2k_super overconfident)
}
PROXY_MIX: dict[str, str | None] = {
    "IQ2_XS": None,
    "IQ2_M": "IQ2_M",      # routes v_proj/o_proj/down-first-eighth -> iq3_s proxy
    "Q2_K_S": None,
}


# ---- the 11 rows -----------------------------------------------------------

@dataclass(frozen=True)
class Row:
    label: str       # short tag for paths
    source: str      # "vanilla" | "qat"
    quant: str       # IQ2_XS | IQ2_M | Q2_K_S | Q2_K
    method: str      # "plain" | "imatrix" | "awq"

    @property
    def use_imatrix(self) -> bool:
        return self.method in ("imatrix", "awq")

    @property
    def gguf_name(self) -> str:
        stem = ("gemma-4-31B-it" if self.source == "vanilla"
                else "gemma-4-31B-it-qat") + f"-{self.quant}"
        if self.method == "plain":
            return f"{stem}-plain.gguf"
        if self.method == "imatrix":
            return f"{stem}-imatrix.gguf"
        return f"{stem}-awq-cv-gate.gguf"


ROWS: tuple[Row, ...] = (
    Row("v_q2k_plain", "vanilla", "Q2_K",   "plain"),
    Row("v_iq2xs_im",  "vanilla", "IQ2_XS", "imatrix"),
    Row("v_iq2xs_awq", "vanilla", "IQ2_XS", "awq"),
    Row("v_iq2m_im",   "vanilla", "IQ2_M",  "imatrix"),
    Row("v_iq2m_awq",  "vanilla", "IQ2_M",  "awq"),
    Row("v_q2ks_im",   "vanilla", "Q2_K_S", "imatrix"),
    Row("v_q2ks_awq",  "vanilla", "Q2_K_S", "awq"),
    Row("q_iq2xs_im",  "qat",     "IQ2_XS", "imatrix"),
    Row("q_iq2xs_awq", "qat",     "IQ2_XS", "awq"),
    Row("q_q2ks_im",   "qat",     "Q2_K_S", "imatrix"),
    Row("q_q2ks_awq",  "qat",     "Q2_K_S", "awq"),
)


# ---- helpers ---------------------------------------------------------------

def _src_paths(source: str) -> tuple[Path, Path, Path]:
    if source == "vanilla":
        return VANILLA_HF, VANILLA_F16, VANILLA_IMATRIX
    return QAT_HF, QAT_F16, QAT_IMATRIX


def _csv_has_qpath(csv_path: Path, qpath: Path) -> bool:
    if not csv_path.exists():
        return False
    needle = str(qpath)
    with csv_path.open() as fh:
        return any(needle in line for line in fh)


def _bench(qpath: Path, label: str, n_params: int, csv_path: Path, log_dir: Path) -> None:
    if _csv_has_qpath(csv_path, qpath):
        log(f"  {qpath.name}: bench row already in CSV — skipping")
        return
    row = runner.bench_one(
        qpath, label,
        reference_n_params=n_params,
        eval_dataset=EVAL_CORPUS,
        eval_baseline=BASELINE_KLD,
        eval_ctx=EVAL_CTX,
        log_dir=log_dir,
        suite="kld",
    )
    runner.append_row(csv_path, row)
    log(f"  {qpath.name}: PPL={row.ppl:.4f} mean_KLD={row.mean_kld:.5f} "
        f"med_KLD={row.median_kld:.5f} top_p={row.same_top_p:.4f}%")


def _free_awq_intermediates(sub: Path) -> None:
    for child in (sub / "model_awq", sub / "model-f16-awq.gguf"):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        elif child.exists():
            child.unlink(missing_ok=True)


# ---- setup -----------------------------------------------------------------

def _setup_source(source: str, model_id: str) -> None:
    src_hf, src_f16, src_imatrix = _src_paths(source)
    src_hf.parent.mkdir(parents=True, exist_ok=True)
    src_imatrix.parent.mkdir(parents=True, exist_ok=True)
    logs = src_imatrix.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with phase(f"[exp-044][setup] extract {source} HF"):
        if src_hf.is_symlink():
            src_hf.unlink()
        step("extract", src_hf / "config.json",
             lambda: extract.extract_text_lm(source=model_id, output_dir=src_hf))
    with phase(f"[exp-044][setup] convert {source} -> F16 GGUF"):
        step("convert", src_f16,
             lambda: convert.hf_to_f16_gguf(src_hf, src_f16, log=logs / "convert.log"))
    with phase(f"[exp-044][setup] {source} base imatrix on new cal corpus (ctx {IMATRIX_CTX})"):
        step("imatrix", src_imatrix,
             lambda: llama_cpp.imatrix(src_f16, CAL_CORPUS, src_imatrix,
                                       ctx=IMATRIX_CTX, log=logs / "imatrix.log"))


def regen_corpus() -> None:
    """Build the windowed corpus with a Qwopus-style val (logtrain TEST slice)."""
    import importlib.util
    CORPORA.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "build_corpora", REPO / "scripts" / "build_corpora.py")
    assert spec and spec.loader
    bc = importlib.util.module_from_spec(spec)
    sys.modules["build_corpora"] = bc
    spec.loader.exec_module(bc)
    with phase("[exp-044][setup] build windowed corpus (Qwopus val = logtrain TEST)"):
        bc.build(
            out_dir=CORPORA,
            model_dir=VANILLA_HF,           # Gemma tokenizer for chat-templating
            wiki_test=WIKI_TEST,
            cal_tokens=500_000,
            val_tokens=10_000,              # logtrain TEST slice (Qwopus recipe)
            eval_tokens_per_domain=30_000,
            seed=42,
            val_supplement_override=None,   # NO MMMU — pure logtrain val
        )
    import json
    w = json.loads(AUDIT.read_text())["calibration"]["pack_audit"].get("windowing")
    if w:
        log(f"  cal: {w['windows_emitted']} windows / {w['sessions_kept']} sessions, "
            f"tool-turn token share {w['tool_turn_token_share']:.0%}")


def build_baseline() -> None:
    (EXP44 / "logs").mkdir(parents=True, exist_ok=True)
    with phase("[exp-044][setup] vanilla FP16 baseline KLD (eval corpus, rebuilt for new llama)"):
        step("baseline", BASELINE_KLD,
             lambda: kld.build_baseline(VANILLA_F16, EVAL_CORPUS, BASELINE_KLD,
                                        ctx=EVAL_CTX, log=EXP44 / "logs" / "baseline.log"))


def setup() -> None:
    EXP44.mkdir(parents=True, exist_ok=True)
    # vanilla first (its tokenizer builds the corpus the imatrices then consume).
    VANILLA_DIR.mkdir(parents=True, exist_ok=True)
    (VANILLA_DIR / "logs").mkdir(parents=True, exist_ok=True)
    with phase("[exp-044][setup] extract vanilla HF"):
        step("extract", VANILLA_HF / "config.json",
             lambda: extract.extract_text_lm(source=VANILLA_ID, output_dir=VANILLA_HF))
    with phase("[exp-044][setup] convert vanilla -> F16 GGUF"):
        step("convert", VANILLA_F16,
             lambda: convert.hf_to_f16_gguf(VANILLA_HF, VANILLA_F16,
                                            log=VANILLA_DIR / "logs" / "convert.log"))
    regen_corpus()
    with phase(f"[exp-044][setup] vanilla base imatrix (ctx {IMATRIX_CTX})"):
        step("imatrix", VANILLA_IMATRIX,
             lambda: llama_cpp.imatrix(VANILLA_F16, CAL_CORPUS, VANILLA_IMATRIX,
                                       ctx=IMATRIX_CTX,
                                       log=VANILLA_DIR / "logs" / "imatrix.log"))
    build_baseline()
    _setup_source("qat", QAT_ID)
    log("=== exp-044 setup complete — inputs ready for the sweep / 11-row run ===")


# ---- row dispatch ----------------------------------------------------------

def run_imatrix_or_plain(row: Row, *, csv_path: Path) -> None:
    _src_hf, src_f16, imatrix = _src_paths(row.source)
    sub = EXP44 / row.label
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    qpath = sub / row.gguf_name
    n_params = bpw_mod.n_params(src_f16)
    tag = f"[exp-044][{row.label}]"

    with phase(f"{tag} quantize {row.quant} ({row.method})"):
        step("quantize", qpath,
             lambda: gguf.quantize(
                 src_f16, qpath, row.quant,
                 imatrix=(imatrix if row.use_imatrix else None),
                 log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        method_tag = {"plain": "plain", "imatrix": "imatrix",
                      "awq": "awq-cv-gate+imatrix"}[row.method]
        model_id = VANILLA_ID if row.source == "vanilla" else QAT_ID
        label = f"{model_id}|{row.quant}|{method_tag} // baseline=vanilla-FP16"
        _bench(qpath, label, n_params, csv_path, log_dir)


def run_awq(row: Row, *, csv_path: Path) -> None:
    src_hf, src_f16, imatrix = _src_paths(row.source)
    sub = EXP44 / row.label
    log_dir = sub / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bundle = sub / "awq.pt"
    model_awq = sub / "model_awq"
    f16_awq = sub / "model-f16-awq.gguf"
    qpath = sub / row.gguf_name
    n_params = bpw_mod.n_params(src_f16)

    proxy = PROXY[row.quant] or awq.proxy_for_quant_type(row.quant)
    proxy_mix = PROXY_MIX[row.quant]
    tag = f"[exp-044][{row.label}]"

    with phase(f"{tag} AWQ calibrate (proxy={proxy}, mix={proxy_mix})"):
        step("AWQ calibrate", bundle,
             lambda: awq.calibrate(
                 src_hf, CAL_CORPUS, bundle,
                 proxy=proxy,
                 proxy_mix=proxy_mix,
                 proxy_tokens=PROXY_TOKENS,
                 ctx=CTX,
                 device="auto",
                 dtype="bfloat16",
                 per_tensor_alpha=True,
                 per_tensor_grid_radius=PER_TENSOR_GRID_RADIUS,
                 holdout_text=VAL_CORPUS,
                 cv_strategy="gate",
                 cv_weight=1.0,
             ))

    with phase(f"{tag} AWQ apply"):
        step("AWQ apply", model_awq / "config.json",
             lambda: awq.apply(
                 src_hf, bundle, model_awq,
                 device="auto",
                 dtype="bfloat16",
                 rmsnorm_plus_one=RMSNORM_PLUS_ONE,
                 sanity_max_rel=SANITY_MAX_REL,
             ))

    with phase(f"{tag} convert AWQ-folded HF -> F16 GGUF"):
        step("convert", f16_awq,
             lambda: convert.hf_to_f16_gguf(model_awq, f16_awq,
                                            log=log_dir / "convert.log"))

    # Faithful to exp-034: quantize the folded F16 with the UNFOLDED base imatrix.
    with phase(f"{tag} quantize {row.quant}"):
        step("quantize", qpath,
             lambda: gguf.quantize(f16_awq, qpath, row.quant, imatrix=imatrix,
                                   log=log_dir / "quantize.log"))

    with phase(f"{tag} bench"):
        model_id = VANILLA_ID if row.source == "vanilla" else QAT_ID
        label = (f"{model_id}|{row.quant}|awq-cv-gate+imatrix|"
                 f"proxy={proxy},mix={proxy_mix},tokens={PROXY_TOKENS},ctx={CTX} "
                 f"// baseline=vanilla-FP16")
        _bench(qpath, label, n_params, csv_path, log_dir)

    _free_awq_intermediates(sub)


def quants() -> int:
    required = [VANILLA_HF / "config.json", VANILLA_F16, VANILLA_IMATRIX,
                QAT_HF / "config.json", QAT_F16, QAT_IMATRIX,
                CAL_CORPUS, VAL_CORPUS, EVAL_CORPUS, BASELINE_KLD]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "exp-044 quants: run `setup` first — missing:\n  " + "\n  ".join(missing))
    EXP44.mkdir(parents=True, exist_ok=True)
    csv_path = EXP44 / "results.csv"
    for row in ROWS:
        if row.method == "awq":
            run_awq(row, csv_path=csv_path)
        else:
            run_imatrix_or_plain(row, csv_path=csv_path)
    log("")
    log("=== exp-044 quants complete (11 rows; QAT IQ2_M skipped) ===")
    log(f"  results: {csv_path}")
    return 0


# ---- publish ---------------------------------------------------------------

def publish() -> None:
    RELEASE_CAL.mkdir(parents=True, exist_ok=True)
    with phase("[exp-044][publish] swap GGUFs + corpus into the release dir"):
        for row in ROWS:
            src = EXP44 / row.label / row.gguf_name
            if not src.exists():
                log(f"  WARN missing {src} — skipping")
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
