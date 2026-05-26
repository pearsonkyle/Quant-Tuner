"""Experiment 004: imatrix combiner sweep on Jackrong/Qwopus3.5-9B-Coder.

Holds the input signals fixed (E[a²] and ‖W[:,c]‖²·E[a²]) and sweeps the
combiner / normalization. See README.md for the cell definitions.

All five cells reuse exp-001's vanilla custom imatrix as the base — the
only thing that varies per cell is how `_combine` blends the two signals.
Output: out/exp-004/...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate.imatrix import (
    _col_l2_sq,
    _l1_normalize,
    _load_base_imatrix,
    _load_weights,
    write_imatrix,
)
from quant_tuner.experiments import log, phase, step
from quant_tuner.models.hf_gguf_map import is_ssm
from quant_tuner.quantize import gguf


MODEL = "Jackrong/Qwopus3.5-9B-Coder"
SLUG = MODEL.replace("/", "__")

EXP1 = REPO / "out" / "exp-001" / SLUG
F16 = EXP1 / "model-f16.gguf"
BASE_IMATRIX = EXP1 / "imatrix-custom.gguf"
EVAL_DS = EXP1 / "corpus.eval.txt"
BASE_KLD = EXP1 / "baseline.kld"

EXP4_ROOT = REPO / "out" / "exp-004"
WORK = EXP4_ROOT / SLUG
LOGS = WORK / "logs"
AGGREGATE_CSV = EXP4_ROOT / "results.csv"


# --------------------------------------------------------------------------- #
# Per-cell combiners. Each takes (e_a2, wcol_l2_sq) -> per-input-channel score.
# All operate per tensor; SSM tensors handled separately (see _build_one).
# --------------------------------------------------------------------------- #

def _alpha_blend(alpha: float):
    def combine(e: np.ndarray, w: np.ndarray) -> np.ndarray:
        a = _l1_normalize(e)
        b = _l1_normalize(w * e)
        return alpha * a + (1.0 - alpha) * b
    return combine


def _mix_50(e: np.ndarray, w: np.ndarray) -> np.ndarray:
    a = _l1_normalize(e)
    b = _l1_normalize(w * e)
    return np.sqrt(np.maximum(a, 0.0) * np.maximum(b, 0.0))


def _max_no_norm(e: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`max(E[a²], ‖W‖²·E[a²])` — keep natural scale; output term will dominate."""
    return np.maximum(e.astype(np.float32), (w * e).astype(np.float32))


CELLS = {
    "C1": ("mix_50",         _mix_50),
    "C2": ("alpha_0.25",     _alpha_blend(0.25)),
    "C3": ("alpha_0.50",     _alpha_blend(0.50)),
    "C4": ("alpha_0.75",     _alpha_blend(0.75)),
    "C5": ("max_no_norm",    _max_no_norm),
}


# --------------------------------------------------------------------------- #
# Build one imatrix using a chosen combiner. Pattern mirrors
# `calibrate.imatrix._build_simple` (the path for analytic variants).
# --------------------------------------------------------------------------- #

def _build_one(combine, out_path: Path, label: str) -> None:
    base = _load_base_imatrix(BASE_IMATRIX)
    weights = _load_weights(F16)
    importance: dict[str, np.ndarray] = {}
    ssm_n = skipped = 0
    for tname, ea2 in base.items():
        if tname not in weights:
            skipped += 1
            continue
        if is_ssm(tname):
            importance[tname] = ea2.astype(np.float32)
            ssm_n += 1
            continue
        wcol = _col_l2_sq(weights[tname])
        if wcol.shape != ea2.shape:
            skipped += 1
            continue
        importance[tname] = combine(ea2.astype(np.float32), wcol.astype(np.float32)).astype(np.float32)
    print(f"[{label}] {len(importance)} tensors ({ssm_n} SSM passthrough, {skipped} skipped)",
          file=sys.stderr)
    write_imatrix(out_path, importance, BASE_IMATRIX, dataset_label=f"<quant-tuner-exp004-{label}>")


# --------------------------------------------------------------------------- #
# Per-cell pipeline
# --------------------------------------------------------------------------- #

def _run_cell(name: str, *, dry_run: bool, n_params: int) -> None:
    label, combine = CELLS[name]
    imatrix_path = WORK / f"imatrix-{name}.gguf"
    quant_path = WORK / f"Q4_K_M-{name}.gguf"

    with phase(f"cell {name} ({label})"):
        if dry_run:
            log(f"  [dry-run] imatrix={imatrix_path.name} quant={quant_path.name}")
            return

        step(f"build {label} imatrix", imatrix_path,
             lambda c=combine, p=imatrix_path, lab=label: _build_one(c, p, lab))

        step(f"quantize Q4_K_M ({name})", quant_path,
             lambda q=quant_path, i=imatrix_path: gguf.quantize(
                 F16, q, "Q4_K_M", imatrix=i,
                 log=LOGS / f"quantize-{name}.log"))

        bench_label = f"{MODEL}|imatrix|custom+{label}"
        with phase(f"bench {bench_label}"):
            row = runner.bench_one(
                quant_path, bench_label,
                reference_n_params=n_params,
                eval_dataset=EVAL_DS, eval_baseline=BASE_KLD,
                eval_ctx=8192, log_dir=LOGS, suite="kld",
            )
            runner.append_row(WORK / "results.csv", row)
            runner.append_row(AGGREGATE_CSV, row)
            log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                f"ppl={row.ppl} mean_kld={row.mean_kld} "
                f"same_top_p={row.same_top_p}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only these cells (C1..C5). Default: all.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for required in (F16, BASE_IMATRIX, EVAL_DS, BASE_KLD):
        if not required.exists():
            print(f"missing prerequisite: {required}", file=sys.stderr)
            return 2

    cells = args.only if args.only else list(CELLS.keys())
    unknown = set(cells) - set(CELLS)
    if unknown:
        print(f"unknown cell(s): {sorted(unknown)} (valid: {sorted(CELLS)})", file=sys.stderr)
        return 2

    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(F16) if not args.dry_run else 0

    with phase(f"exp-004 ({len(cells)} cells)"):
        for name in cells:
            _run_cell(name, dry_run=args.dry_run, n_params=n_params)

    log("ALL DONE")
    log("next: run scripts/run_toolcall_reps.py per REPRODUCE.md step 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
