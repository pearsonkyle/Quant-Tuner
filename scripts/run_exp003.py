"""Experiment 003: imatrix token & sequence-length sensitivity.

11 cells, all Jackrong/Qwopus3.5-9B-Coder, vanilla `E[a²]` re-weighting.
See experiments/003-imatrix-token-sensitivity/README.md for the full
matrix and rationale.

Reuses exp-001's F16 GGUF, eval corpus, and F16 KLD baseline. New
artifacts go to out/exp-003/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.data import ingest, split
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import llama_cpp
from quant_tuner.quantize import gguf


MODEL = "Jackrong/Qwopus3.5-9B-Coder"
SLUG = MODEL.replace("/", "__")

EXP1 = REPO / "out" / "exp-001" / SLUG
F16 = EXP1 / "model-f16.gguf"
MODEL_DIR = EXP1 / "model_extracted"
EVAL_DS = EXP1 / "corpus.eval.txt"
BASE_KLD = EXP1 / "baseline.kld"
WIKI = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"
LOGTRAIN = REPO / "logtrain.jsonl"
SUPPLEMENT = REPO / "calibration_supplement.txt"

EXP3_ROOT = REPO / "out" / "exp-003"
WORK = EXP3_ROOT / SLUG
LOGS = WORK / "logs"
AGGREGATE_CSV = EXP3_ROOT / "results.csv"


@dataclass(frozen=True)
class Cell:
    name: str          # A1, B11, …
    sub: str           # "A" or "B"
    dataset: str       # "custom" | "wiki" | "combined"
    tokens: int | None # target token count for custom; None for wiki
    ctx: int           # llama-imatrix sequence length

    @property
    def corpus_path(self) -> Path:
        if self.dataset == "wiki":
            return WIKI
        if self.dataset == "combined":
            return WORK / "corpus.combined.txt"
        return WORK / f"corpus.custom_{self.tokens // 1000}K.txt"

    @property
    def imatrix_path(self) -> Path:
        return WORK / f"imatrix.{self.name}.gguf"

    @property
    def quant_path(self) -> Path:
        return WORK / f"Q4_K_M.{self.name}.gguf"

    @property
    def label(self) -> str:
        ds = self.dataset
        tok = "—" if self.dataset == "wiki" else f"{self.tokens//1000}K"
        return f"{MODEL}|{self.sub}/{self.name}|{ds}|tok={tok}|ctx={self.ctx//1024}K"


# Sub-experiment A: dataset axis at ctx=8K
A_CELLS = [
    Cell("A1", "A", "custom",   500_000, 8192),
    Cell("A2", "A", "wiki",     None,    8192),
    Cell("A3", "A", "combined", 500_000, 8192),
]

# Sub-experiment B: custom × tokens × ctx grid
B_CELLS = [
    Cell(f"B{r}{c}", "B", "custom", tokens, ctx)
    for r, tokens in enumerate([100_000, 250_000, 500_000], start=1)
    for c, ctx     in enumerate([8192,   16384,  20480 ], start=1)
]

# B31 == A1 (custom 500K, ctx=8K). Dedupe by writing one row under A1's name;
# B31 reuses A1's artifacts.
ALL_CELLS: list[Cell] = A_CELLS + [c for c in B_CELLS if not (c.tokens == 500_000 and c.ctx == 8192)]
B31_ALIAS_OF = "A1"  # render_exp003_tables.py will duplicate this row under B31


# --------------------------------------------------------------------------- #
# Corpus prep
# --------------------------------------------------------------------------- #

def _prepare_custom_corpus(target_tokens: int, out_path: Path) -> None:
    """Build the custom calibration corpus at target_tokens, identical pipeline
    to exp-001 (filter, seed=42 split, stratified_pack on train slice, supplement
    appended). Different target_tokens use the same session pool — the smaller
    sizes are deterministic subsets, not new random draws.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, fix_mistral_regex=True)
    sessions = ingest.load_sessions(LOGTRAIN)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
    chunks, _kept, total, audit = split.stratified_pack(
        splits["train"], tok,
        target_tokens=target_tokens, per_session_cap=6_000, seed=42,
    )
    supplement = SUPPLEMENT if SUPPLEMENT.exists() else None
    split.write_corpus(chunks, out_path, supplement=supplement)
    (out_path.with_suffix(".audit.json")).write_text(
        json.dumps(audit, indent=2, default=str)
    )
    log(f"  custom corpus: {total:,} tokens (+ supplement) -> {out_path.name}")


def _prepare_combined_corpus(custom_path: Path, out_path: Path) -> None:
    """Concatenate the 500K custom corpus + wiki.test.raw verbatim."""
    with out_path.open("wb") as f, custom_path.open("rb") as a, WIKI.open("rb") as b:
        shutil.copyfileobj(a, f)
        f.write(b"\n")
        shutil.copyfileobj(b, f)
    log(f"  combined corpus: {custom_path.stat().st_size + WIKI.stat().st_size:,} bytes -> {out_path.name}")


# --------------------------------------------------------------------------- #
# Per-cell pipeline
# --------------------------------------------------------------------------- #

def _run_cell(cell: Cell, *, dry_run: bool, n_params: int) -> None:
    with phase(f"cell {cell.name} ({cell.dataset}, tokens={cell.tokens}, ctx={cell.ctx})"):
        if dry_run:
            log(f"  [dry-run] corpus={cell.corpus_path.name} "
                f"imatrix={cell.imatrix_path.name} quant={cell.quant_path.name}")
            return

        if cell.dataset == "custom":
            step(f"prepare custom corpus ({cell.tokens//1000}K)", cell.corpus_path,
                 lambda: _prepare_custom_corpus(cell.tokens, cell.corpus_path))
        elif cell.dataset == "combined":
            # combined depends on the 500K custom corpus
            custom_500k = WORK / "corpus.custom_500K.txt"
            step("prepare custom corpus (500K) for combined", custom_500k,
                 lambda: _prepare_custom_corpus(500_000, custom_500k))
            step("prepare combined corpus (custom_500K + wiki)", cell.corpus_path,
                 lambda: _prepare_combined_corpus(custom_500k, cell.corpus_path))
        # wiki: no prep — uses WIKI directly

        step(f"llama-imatrix ({cell.dataset}, ctx={cell.ctx})", cell.imatrix_path,
             lambda c=cell: llama_cpp.imatrix(
                 F16, c.corpus_path, c.imatrix_path,
                 ctx=c.ctx, log=LOGS / f"imatrix.{c.name}.log"))

        step(f"quantize Q4_K_M ({cell.name})", cell.quant_path,
             lambda c=cell: gguf.quantize(
                 F16, c.quant_path, "Q4_K_M", imatrix=c.imatrix_path,
                 log=LOGS / f"quantize.{c.name}.log"))

        with phase(f"bench {cell.label}"):
            row = runner.bench_one(
                cell.quant_path, cell.label,
                reference_n_params=n_params,
                eval_dataset=EVAL_DS,
                eval_baseline=BASE_KLD,
                eval_ctx=8192,
                log_dir=LOGS,
                suite="kld",
            )
            runner.append_row(WORK / "results.csv", row)
            runner.append_row(AGGREGATE_CSV, row)
            log(f"  size={row.size_gib:.2f} GiB bpw={row.bpw:.3f} "
                f"ppl={row.ppl} mean_kld={row.mean_kld} "
                f"same_top_p={row.same_top_p}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only these cell names (e.g. A1 B11 B22). "
                         "If unspecified, runs all 11.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned cells without running.")
    args = ap.parse_args()

    for required in (F16, MODEL_DIR, EVAL_DS, BASE_KLD, WIKI):
        if not required.exists():
            print(f"missing prerequisite: {required}", file=sys.stderr)
            return 2

    if args.only:
        wanted = set(args.only)
        selected = [c for c in ALL_CELLS if c.name in wanted]
        unknown = wanted - {c.name for c in ALL_CELLS}
        if unknown:
            print(f"unknown cell(s): {sorted(unknown)} (B31 is an alias of A1)",
                  file=sys.stderr)
            return 2
    else:
        selected = ALL_CELLS

    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    n_params = bpw_mod.n_params(F16) if not args.dry_run else 0

    with phase(f"exp-003 ({len(selected)} cells)"):
        for cell in selected:
            _run_cell(cell, dry_run=args.dry_run, n_params=n_params)

    log("ALL DONE")
    log(f"aggregate CSV: {AGGREGATE_CSV.relative_to(REPO)}")
    log("render tables: PYTHONPATH=src .venv/bin/python scripts/render_exp003_tables.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
