"""Experiment 006: outlier_max rescue + heavy-tail signal mixing.

7 cells. All variants rerank the existing exp-002 forward_stats.npz +
exp-001 vanilla base imatrix; no new HF forward pass. See README.md.

Cell taxonomy:
  F1, F2, F3 — outlier_max with per-tensor-class fallback to a milder
               signal on `attn_qkv` and/or `attn_gate` (the suspected
               culprits of outlier_max's regression).
  H1, H2, H3 — combiners over (√E[a⁴], max|a|): max, blend, geometric.
  H4         — combine vanilla E[a²] with outlier_l4's √E[a⁴].
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate.imatrix import (
    ForwardStats,
    _l1_normalize,
    _load_base_imatrix,
    write_imatrix,
)
from quant_tuner.experiments import log, phase, step
from quant_tuner.models.hf_gguf_map import is_ssm
from quant_tuner.quantize import gguf


MODEL = "Jackrong/Qwopus3.5-9B-Coder"
SLUG = MODEL.replace("/", "__")

EXP1 = REPO / "out" / "exp-001" / SLUG
EXP2 = REPO / "out" / "exp-002" / SLUG
F16 = EXP1 / "model-f16.gguf"
BASE_IMATRIX = EXP1 / "imatrix-custom.gguf"   # provides E[a²]
FORWARD_STATS = EXP2 / "forward_stats.npz"    # provides E[a⁴], max|a|
EVAL_DS = EXP1 / "corpus.eval.txt"
BASE_KLD = EXP1 / "baseline.kld"

EXP6_ROOT = REPO / "out" / "exp-006"
WORK = EXP6_ROOT / SLUG
LOGS = WORK / "logs"
AGGREGATE_CSV = EXP6_ROOT / "results.csv"

_CLASS_RE = re.compile(r"^blk\.\d+\.(?P<cls>[^.]+)\.weight$")


def _classify(name: str) -> str:
    m = _CLASS_RE.match(name)
    return m.group("cls") if m else ""


# --------------------------------------------------------------------------- #
# Per-tensor scorer signature: (e_a2, sqrt_e_a4, max_abs) -> per-channel score
# --------------------------------------------------------------------------- #

Scorer = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _outlier_max(e_a2, l4, mx):
    return mx


def _outlier_l4(e_a2, l4, mx):
    return l4


def _vanilla(e_a2, l4, mx):
    return e_a2


def _max_l4_mx(e_a2, l4, mx):
    return np.maximum(_l1_normalize(l4), _l1_normalize(mx))


def _blend_l4_mx(e_a2, l4, mx):
    return 0.5 * _l1_normalize(l4) + 0.5 * _l1_normalize(mx)


def _geom_l4_mx(e_a2, l4, mx):
    a = _l1_normalize(l4)
    b = _l1_normalize(mx)
    return np.sqrt(np.maximum(a, 0.0) * np.maximum(b, 0.0))


def _max_e2_l4(e_a2, l4, mx):
    return np.maximum(_l1_normalize(e_a2), _l1_normalize(l4))


@dataclass(frozen=True)
class Cell:
    name: str
    description: str
    primary: Scorer
    fallback: Scorer | None = None
    fallback_classes: tuple[str, ...] = ()


CELLS: list[Cell] = [
    Cell("F1", "outlier_max with E[a²] on attn_qkv + attn_gate",
         _outlier_max, _vanilla, ("attn_qkv", "attn_gate")),
    Cell("F2", "outlier_max with outlier_l4 on attn_qkv + attn_gate",
         _outlier_max, _outlier_l4, ("attn_qkv", "attn_gate")),
    # F3's fallback class set is the same as F1's here — but the docs
    # call out broader linear_attn coverage. On this model the only
    # non-SSM linear_attn outputs are attn_qkv and attn_gate; ssm_*
    # ones are filtered by is_ssm. If a different model surfaces more,
    # extend this tuple.
    Cell("F3", "outlier_max with E[a²] on all linear_attn-mapped tensors",
         _outlier_max, _vanilla, ("attn_qkv", "attn_gate")),
    Cell("H1", "max(L1(√E[a⁴]), L1(max|a|))", _max_l4_mx),
    Cell("H2", "0.5·L1(√E[a⁴]) + 0.5·L1(max|a|)", _blend_l4_mx),
    Cell("H3", "sqrt(L1(√E[a⁴]) · L1(max|a|))", _geom_l4_mx),
    Cell("H4", "max(L1(E[a²]), L1(√E[a⁴]))", _max_e2_l4),
]


# --------------------------------------------------------------------------- #
# Imatrix construction
# --------------------------------------------------------------------------- #

def _build_imatrix(cell: Cell, out_path: Path) -> None:
    base = _load_base_imatrix(BASE_IMATRIX)          # {gguf_name: E[a²]}
    stats = ForwardStats.load(FORWARD_STATS)         # E[a⁴], max|a|
    importance: dict[str, np.ndarray] = {}
    ssm_n = primary_n = fallback_n = skipped = 0

    for tname, e_a2 in base.items():
        if is_ssm(tname):
            importance[tname] = e_a2.astype(np.float32)
            ssm_n += 1
            continue
        e_a4 = stats.e_a4.get(tname)
        mx = stats.max_abs.get(tname)
        if e_a4 is None or mx is None or e_a4.shape != e_a2.shape:
            # No forward stats → fall back to E[a²] passthrough
            importance[tname] = e_a2.astype(np.float32)
            skipped += 1
            continue
        l4 = np.sqrt(np.maximum(e_a4, 0.0)).astype(np.float32)

        cls = _classify(tname)
        if cell.fallback is not None and cls in cell.fallback_classes:
            score = cell.fallback(e_a2.astype(np.float32), l4, mx.astype(np.float32))
            fallback_n += 1
        else:
            score = cell.primary(e_a2.astype(np.float32), l4, mx.astype(np.float32))
            primary_n += 1
        importance[tname] = score.astype(np.float32)

    print(f"[{cell.name}] {len(importance)} tensors "
          f"({primary_n} primary, {fallback_n} fallback, "
          f"{ssm_n} SSM passthrough, {skipped} no-stats fallback)",
          file=sys.stderr)
    write_imatrix(
        out_path, importance, BASE_IMATRIX,
        dataset_label=f"<quant-tuner-exp006-{cell.name}>",
    )


def _run_cell(cell: Cell, *, dry_run: bool, n_params: int) -> None:
    imatrix_path = WORK / f"imatrix-{cell.name}.gguf"
    quant_path = WORK / f"Q4_K_M-{cell.name}.gguf"

    with phase(f"cell {cell.name} ({cell.description})"):
        if dry_run:
            log(f"  [dry-run] imatrix={imatrix_path.name} quant={quant_path.name}")
            return

        step(f"build imatrix {cell.name}", imatrix_path,
             lambda c=cell, p=imatrix_path: _build_imatrix(c, p))

        step(f"quantize Q4_K_M ({cell.name})", quant_path,
             lambda q=quant_path, i=imatrix_path: gguf.quantize(
                 F16, q, "Q4_K_M", imatrix=i,
                 log=LOGS / f"quantize-{cell.name}.log"))

        bench_label = f"{MODEL}|imatrix|custom+exp006_{cell.name}"
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
                    help=f"Cells to run (subset of {sorted(c.name for c in CELLS)})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for required in (F16, BASE_IMATRIX, FORWARD_STATS, EVAL_DS, BASE_KLD):
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

    with phase(f"exp-006 ({len(selected)} cells)"):
        for cell in selected:
            _run_cell(cell, dry_run=args.dry_run, n_params=n_params)

    log("ALL DONE")
    log("next: scripts/run_toolcall_reps.py per REPRODUCE.md step 5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
