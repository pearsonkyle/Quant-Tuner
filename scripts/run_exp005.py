"""Experiment 005: block-aware imatrix aggregation on Jackrong/Qwopus3.5-9B-Coder.

Post-processes an existing imatrix by aggregating per-channel scores into
groups matching Q4_K_M's super-block size (256 input channels), then
broadcasting the aggregated value back across the block. See README.md.

All five cells reuse existing imatrices from exp-001 and exp-002 —
no llama-imatrix passes, no forward stats. Aggregation is numpy-only.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate.imatrix import _load_base_imatrix, write_imatrix
from quant_tuner.experiments import log, phase, step
from quant_tuner.models.hf_gguf_map import is_ssm
from quant_tuner.quantize import gguf


MODEL = "Jackrong/Qwopus3.5-9B-Coder"
SLUG = MODEL.replace("/", "__")

EXP1 = REPO / "out" / "exp-001" / SLUG
EXP2 = REPO / "out" / "exp-002" / SLUG
F16 = EXP1 / "model-f16.gguf"
BASE_VANILLA = EXP1 / "imatrix-custom.gguf"
BASE_OUTPUT_AWARE = EXP2 / "imatrix-output_aware.gguf"
EVAL_DS = EXP1 / "corpus.eval.txt"
BASE_KLD = EXP1 / "baseline.kld"

EXP5_ROOT = REPO / "out" / "exp-005"
WORK = EXP5_ROOT / SLUG
LOGS = WORK / "logs"
AGGREGATE_CSV = EXP5_ROOT / "results.csv"


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Cell:
    name: str
    base_path: Path
    base_label: str
    agg: str       # "max" | "mean"
    block: int


CELLS: list[Cell] = [
    Cell("E1", BASE_VANILLA,       "vanilla",      "max",  256),
    Cell("E2", BASE_VANILLA,       "vanilla",      "mean", 256),
    Cell("E3", BASE_OUTPUT_AWARE,  "output_aware", "max",  256),
    Cell("E4", BASE_OUTPUT_AWARE,  "output_aware", "mean", 256),
    Cell("E5", BASE_VANILLA,       "vanilla",      "max",  32),
]


# --------------------------------------------------------------------------- #
# Block aggregation
# --------------------------------------------------------------------------- #

def _block_aggregate(scores: np.ndarray, *, agg: str, block: int) -> np.ndarray:
    """Reshape per-channel scores into groups of `block`, aggregate, broadcast back.

    If n_in is not a multiple of `block`, the tail is aggregated on its own
    (so the function never raises) — but tails should be rare for Q4_K_M
    layers, where n_in is typically a multiple of 256.
    """
    n = scores.shape[0]
    n_full = (n // block) * block
    head = scores[:n_full].reshape(-1, block)
    if agg == "max":
        head_agg = head.max(axis=1, keepdims=True)
    elif agg == "mean":
        head_agg = head.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown agg: {agg}")
    head_out = np.broadcast_to(head_agg, head.shape).reshape(-1)

    if n_full == n:
        return head_out.astype(np.float32)

    tail = scores[n_full:]
    tail_val = tail.max() if agg == "max" else tail.mean()
    tail_out = np.full(tail.shape, tail_val, dtype=np.float32)
    return np.concatenate([head_out, tail_out]).astype(np.float32)


def _build_one(cell: Cell, out_path: Path) -> None:
    base = _load_base_imatrix(cell.base_path)
    importance: dict[str, np.ndarray] = {}
    ssm_n = 0
    for tname, scores in base.items():
        if is_ssm(tname):
            importance[tname] = scores.astype(np.float32)
            ssm_n += 1
            continue
        importance[tname] = _block_aggregate(
            scores.astype(np.float32), agg=cell.agg, block=cell.block,
        )
    print(f"[{cell.name}] {len(importance)} tensors aggregated "
          f"({ssm_n} SSM passthrough, agg={cell.agg}, block={cell.block})",
          file=sys.stderr)
    write_imatrix(
        out_path, importance, cell.base_path,
        dataset_label=f"<quant-tuner-exp005-{cell.name}>",
    )


# --------------------------------------------------------------------------- #
# Per-cell pipeline
# --------------------------------------------------------------------------- #

def _run_cell(cell: Cell, *, dry_run: bool, n_params: int) -> None:
    imatrix_path = WORK / f"imatrix-{cell.name}.gguf"
    quant_path = WORK / f"Q4_K_M-{cell.name}.gguf"

    with phase(f"cell {cell.name} ({cell.base_label}, {cell.agg}/{cell.block})"):
        if dry_run:
            log(f"  [dry-run] base={cell.base_path.name} "
                f"imatrix={imatrix_path.name} quant={quant_path.name}")
            return

        step(f"build block-{cell.agg}({cell.block}) imatrix", imatrix_path,
             lambda c=cell, p=imatrix_path: _build_one(c, p))

        step(f"quantize Q4_K_M ({cell.name})", quant_path,
             lambda q=quant_path, i=imatrix_path: gguf.quantize(
                 F16, q, "Q4_K_M", imatrix=i,
                 log=LOGS / f"quantize-{cell.name}.log"))

        bench_label = f"{MODEL}|imatrix|{cell.base_label}+block_{cell.agg}{cell.block}"
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
                    help="Run only these cells (E1..E5). Default: all.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for required in (F16, BASE_VANILLA, BASE_OUTPUT_AWARE, EVAL_DS, BASE_KLD):
        if not required.exists():
            print(f"missing prerequisite: {required}", file=sys.stderr)
            return 2

    if args.only:
        wanted = set(args.only)
        selected = [c for c in CELLS if c.name in wanted]
        unknown = wanted - {c.name for c in CELLS}
        if unknown:
            print(f"unknown cell(s): {sorted(unknown)}", file=sys.stderr)
            return 2
    else:
        selected = CELLS

    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    n_params = bpw_mod.n_params(F16) if not args.dry_run else 0

    with phase(f"exp-005 ({len(selected)} cells)"):
        for cell in selected:
            _run_cell(cell, dry_run=args.dry_run, n_params=n_params)

    log("ALL DONE")
    log("next: run scripts/run_toolcall_reps.py per REPRODUCE.md step 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
