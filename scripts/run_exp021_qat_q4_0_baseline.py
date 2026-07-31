"""Experiment 021: bench google/gemma-4-31B-it-qat-q4_0-gguf on exp-020 eval.

Adds the Google-published QAT (quantization-aware training) Q4_0 GGUF to the
exp-020 comparison table. Reuses the exp-020 eval corpus and F16 baseline
KLD so the row is directly comparable to the AWQ cv-gate / imatrix-only / plain
rows already published.

Note: QAT-Q4_0 is ~4.5 bpw / ~17 GiB — almost 2× the size of the IQ2/Q2 rows
in the table. It is included as a higher-bit-budget reference point, not as a
peer competitor.

Idempotent via experiments.step(). Downloads ~17 GiB on first run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.experiments import log, phase, step

REPO_ID = "google/gemma-4-31B-it"
QAT_HF_REPO = "google/gemma-4-31B-it-qat-q4_0-gguf"
QAT_FILE = "gemma-4-31B_q4_0-it.gguf"

EXP20 = REPO / "out" / "exp-020" / REPO_ID.replace("/", "__")
EXP21 = REPO / "out" / "exp-021" / REPO_ID.replace("/", "__")
LOGS = EXP21 / "logs"

EVAL_CTX = 4096

EVAL_CORPUS = EXP20 / "corpora" / "corpus.eval.txt"
BASELINE_KLD = EXP20 / "baseline.kld"
SRC_F16 = REPO / "out" / "exp-009" / REPO_ID.replace("/", "__") / "model-f16.gguf"


def _check_inputs() -> None:
    missing = [p for p in (EVAL_CORPUS, BASELINE_KLD, SRC_F16) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "exp-021 reuses exp-020 eval corpus + baseline + exp-009 F16:\n  "
            + names
        )


def _download_qat(dest: Path) -> None:
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(
        repo_id=QAT_HF_REPO,
        filename=QAT_FILE,
        cache_dir=EXP21 / "hf_cache",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(Path(cached).resolve())


def _f16_ppl_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", log_path.read_text())
    return float(m.group(1)) if m else None


def main() -> int:
    _check_inputs()
    EXP21.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    qpath = EXP21 / QAT_FILE
    with phase("download QAT-Q4_0 GGUF"):
        step("hf_hub_download", qpath, lambda: _download_qat(qpath))

    n_params = bpw_mod.n_params(SRC_F16)
    log(f"reference n_params (from exp-009 F16) = {n_params:,.0f}")

    csv_path = EXP21 / "results.csv"
    dataset_label = (
        "wiki+500k-logtrain (cal) / MMMU (val) / code+math+tools (eval)"
    )
    label = f"{REPO_ID}|Q4_0|qat (google)|{dataset_label}"

    with phase("bench QAT-Q4_0 vs exp-020 baseline"):
        row = runner.bench_one(
            qpath, label,
            reference_n_params=n_params,
            eval_dataset=EVAL_CORPUS,
            eval_baseline=BASELINE_KLD,
            eval_ctx=EVAL_CTX,
            log_dir=LOGS,
            suite="kld",
        )
        runner.append_row(csv_path, row)
        log(f"  size={row.size_gib:.2f} GiB  bpw={row.bpw:.3f}")
        log(f"  ppl={row.ppl}  mean_kld={row.mean_kld}  same_top_p={row.same_top_p}")

    fp16_ppl = _f16_ppl_from_log(EXP20 / "logs" / "baseline.log")
    md = EXP21 / "row.md"
    md.write_text(
        "| Q4_0 (QAT, google) | "
        f"{row.size_gib:.2f} | {row.bpw:.3f} | "
        f"{row.ppl:.4f} | {row.mean_kld:.5f} | {row.same_top_p:.4f}% |\n"
        f"\nFP16 reference PPL (for context): {fp16_ppl}\n"
    )
    log(f"row -> {md.relative_to(REPO)}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
