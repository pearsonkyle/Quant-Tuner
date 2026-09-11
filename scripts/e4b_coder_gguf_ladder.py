#!/usr/bin/env python3
"""GGUF ladder for gemma4-e4b-coder: IQ2_M / IQ3_M / IQ4_XS, imatrix-guided.

Stages (idempotent via experiments.step(); resumable):

  setup  : imatrix over the 15M-token calibration corpus, then the bf16
           baseline KLD on BOTH eval corpora
  quants : quantize each rung + bench (bpw, PPL, KLD, top-p agreement, speed)
  report : print the ladder table

Two deliberate choices, both inherited from how this model was built:

**No rung above q4.** The base is the QAT-q4_0-unquantized release, whose weights
are already conditioned for a 4-bit grid. Rungs above that grid measure the
conditioning, not the quantizer, so the ladder stops at IQ4_XS.

**imatrix at ctx 32,768, matching the corpus's pack ctx.** A corpus packed at
one context and read at another does not calibrate on longer trajectories -- it
glues unrelated windows into each context, because the cut already happened at
pack time. The calibration split is packed at 32,768 and is read at 32,768.

KLD is measured on two corpora on purpose: `agentic` is in-distribution (what it
was calibrated on) and `general` is not. Reporting only the first would flatter
every rung.

    PYTHONPATH=src .venv/bin/python scripts/e4b_coder_gguf_ladder.py setup
    PYTHONPATH=src .venv/bin/python scripts/e4b_coder_gguf_ladder.py quants
    PYTHONPATH=src .venv/bin/python scripts/e4b_coder_gguf_ladder.py report
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_tuner.bench import bpw as bpw_mod          # noqa: E402
from quant_tuner.bench import kld, runner             # noqa: E402
from quant_tuner.experiments import log, phase, step  # noqa: E402
from quant_tuner.models import llama_cpp              # noqa: E402
from quant_tuner.quantize import gguf                 # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "out" / "e4b-v65536" / "gguf-ladder"
BF16 = Path("/workspace/models/gguf-e4b-coder/gemma4-e4b-coder-BF16.gguf")
CAL = REPO / "out" / "e4b-v65536" / "cal-15m-ctx32k" / "corpus.cal.txt"
EVALS = {
    "agentic": REPO / "out" / "e4b-v2-corpora-15m" / "corpus.eval.agentic.txt",
    "general": REPO / "out" / "e4b-v2-corpora-15m" / "corpus.eval.general.txt",
}

IMATRIX = WORK / "imatrix-cal-ctx32k.gguf"
IMATRIX_CTX = 32768
EVAL_CTX = 8192

# At or below q4 only -- see the module docstring.
QUANTS = ["IQ4_XS", "IQ3_M", "IQ2_M"]

CSV = WORK / "results.csv"


def qpath(quant: str) -> Path:
    return WORK / quant / f"gemma4-e4b-coder-{quant}.gguf"


def baseline_path(name: str) -> Path:
    return WORK / f"baseline.{name}.kld"


def setup() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    for p in (BF16, CAL, *EVALS.values()):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    with phase(f"[e4b-gguf][setup] imatrix over {CAL.name} at ctx {IMATRIX_CTX}"):
        step("imatrix", IMATRIX,
             lambda: llama_cpp.imatrix(BF16, CAL, IMATRIX, ctx=IMATRIX_CTX,
                                       log=logs / "imatrix.log"))

    for name, corpus in EVALS.items():
        with phase(f"[e4b-gguf][setup] bf16 baseline KLD on {name} (ctx {EVAL_CTX})"):
            step(f"baseline-{name}", baseline_path(name),
                 lambda c=corpus, n=name: kld.build_baseline(
                     BF16, c, baseline_path(n), ctx=EVAL_CTX,
                     log=logs / f"baseline.{n}.log"))

    log("=== setup complete: imatrix + 2 baselines ready ===")
    return 0


def _csv_has(qp: Path) -> bool:
    if not CSV.exists():
        return False
    with open(CSV) as fh:
        return any(qp.name in ",".join(r.values()) for r in csv.DictReader(fh))


def quants() -> int:
    missing = [str(p) for p in (IMATRIX, *(baseline_path(n) for n in EVALS))
               if not p.exists()]
    if missing:
        raise SystemExit("run `setup` first -- missing:\n  " + "\n  ".join(missing))

    n_params = bpw_mod.n_params(BF16)

    for quant in QUANTS:
        qp = qpath(quant)
        log_dir = qp.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        with phase(f"[e4b-gguf][{quant}] quantize (imatrix)"):
            step("quantize", qp,
                 lambda p=qp, q=quant, ld=log_dir: gguf.quantize(
                     BF16, p, q, imatrix=IMATRIX, log=ld / "quantize.log"))

        # ONE row per rung carrying BOTH distributions -- BenchRow has ppl/kld
        # for the general corpus and ppl_tools/mean_kld_tools for the agentic
        # one. Reporting only the in-distribution number would flatter every
        # rung: it is the corpus the imatrix was collected on.
        if _csv_has(qp):
            log(f"  {qp.name}: already benched -- skipping")
            continue
        with phase(f"[e4b-gguf][{quant}] bench (general + agentic)"):
            row = runner.bench_one(
                qp, f"gemma4-e4b-coder|{quant}|imatrix // baseline=bf16",
                reference_n_params=n_params,
                eval_dataset=EVALS["general"], eval_baseline=baseline_path("general"),
                tools_dataset=EVALS["agentic"], tools_baseline=baseline_path("agentic"),
                eval_ctx=EVAL_CTX, log_dir=log_dir, suite="kld")
            runner.append_row(CSV, row)
            log(f"  {quant}: bpw={row.bpw:.3f} PPL={row.ppl} "
                f"medKLD={row.median_kld} toolsKLD={row.mean_kld_tools}")

    log(f"=== quants complete: {CSV} ===")
    return 0


def report() -> int:
    if not CSV.exists():
        raise SystemExit(f"no results at {CSV}")
    with open(CSV) as fh:
        rows = list(csv.DictReader(fh))
    print(f"{'model':52s} {'GiB':>6} {'bpw':>6} {'PPL':>9} {'medKLD':>9} {'top-p%':>7}")
    for r in rows:
        print(f"{r['model'][:52]:52s} {float(r['size_gib']):6.2f} "
              f"{float(r['bpw']):6.3f} "
              f"{float(r['ppl']) if r.get('ppl') else float('nan'):9.4f} "
              f"{float(r['median_kld']) if r.get('median_kld') else float('nan'):9.5f} "
              f"{float(r['same_top_p']) if r.get('same_top_p') else float('nan'):7.3f}")
    return 0


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "all":
        setup(); quants(); report()
    else:
        raise SystemExit({"setup": setup, "quants": quants, "report": report}[stage]())
