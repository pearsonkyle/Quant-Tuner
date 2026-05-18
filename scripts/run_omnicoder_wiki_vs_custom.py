"""Add 3 new rows comparing stock vs hybrid imatrix × custom vs wiki corpus.

Builds on the prior run; reuses the F16 GGUF, the custom-corpus stock imatrix,
the eval baseline, and the existing hybrid_custom + Q4_K_M-none + fp16 rows.

New artifacts:
  * imatrix-wiki.gguf            — stock llama-imatrix on wiki.test.raw
  * imatrix-hybrid_wiki.gguf     — hybrid_custom applied to the wiki imatrix
  * Q4_K_M-stock_custom.gguf     — Q4_K_M quantized with custom-corpus stock imatrix
  * Q4_K_M-stock_wiki.gguf       — Q4_K_M quantized with wiki-corpus stock imatrix
  * Q4_K_M-hybrid_wiki.gguf      — Q4_K_M quantized with hybrid+wiki imatrix

After the new benches the script also renames the existing
`imatrix/Q4_K_M-hybrid` results row to `hybrid/Q4_K_M-custom` for symmetry.
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import runner
from quant_tuner.calibrate import imatrix
from quant_tuner.leaderboard import aggregate
from quant_tuner.models import llama_cpp
from quant_tuner.quantize import gguf

WORK = REPO / "out" / "omnicoder_q4_k_m"
LOGS = WORK / "logs"

F16_GGUF = WORK / "model-f16.gguf"
WIKI_SRC = Path("/Users/kpearson/Programs/ai/llm/omnicoder-gguf-experiment/artifacts/wiki.test.raw")
WIKI_LOCAL = WORK / "wiki.test.raw"

BASE_IMATRIX_CUSTOM = WORK / "imatrix-custom.gguf"
BASE_IMATRIX_WIKI = WORK / "imatrix-wiki.gguf"
HYBRID_IMATRIX_CUSTOM = WORK / "imatrix-hybrid_custom.gguf"
HYBRID_IMATRIX_WIKI = WORK / "imatrix-hybrid_wiki.gguf"

QUANT_STOCK_CUSTOM = WORK / "Q4_K_M-stock_custom.gguf"
QUANT_STOCK_WIKI = WORK / "Q4_K_M-stock_wiki.gguf"
QUANT_HYBRID_WIKI = WORK / "Q4_K_M-hybrid_wiki.gguf"

CORPUS_EVAL = WORK / "corpus.eval.txt"
EVAL_BASELINE = WORK / "baseline.kld"
RESULTS = WORK / "results.csv"
LEADERBOARD = WORK / "LEADERBOARD.md"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def phase(name: str):
    class _P:
        def __enter__(self_):
            self_.t0 = time.time()
            log(f"┌─ {name}")
            return self_

        def __exit__(self_, *exc):
            dt = (time.time() - self_.t0) / 60
            log(f"└─ {name} done ({dt:.1f} min)")
            return False
    return _P()


def relabel_row(csv_path: Path, old: str, new: str) -> None:
    """Rename a results.csv row's model column. No-op if old label is absent."""
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open()))
    changed = False
    for r in rows:
        if r["model"] == old:
            r["model"] = new
            changed = True
    if changed:
        fields = rows[0].keys() if rows else runner.CSV_COLUMNS
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fields))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        log(f"  relabeled '{old}' -> '{new}' in {csv_path.name}")


def main() -> int:
    assert F16_GGUF.exists(), f"missing F16 GGUF: {F16_GGUF}"
    assert BASE_IMATRIX_CUSTOM.exists(), f"missing custom imatrix: {BASE_IMATRIX_CUSTOM}"

    # 0) Local copy of wiki.test.raw so the workspace is self-contained.
    if not WIKI_LOCAL.exists():
        shutil.copy(WIKI_SRC, WIKI_LOCAL)
        log(f"copied {WIKI_SRC.name} -> {WIKI_LOCAL}")

    # 1) Stock imatrix on wiki corpus.
    if not BASE_IMATRIX_WIKI.exists():
        with phase("llama-imatrix on wiki.test.raw"):
            llama_cpp.imatrix(F16_GGUF, WIKI_LOCAL, BASE_IMATRIX_WIKI,
                              ctx=512, log=LOGS / "imatrix-wiki.log")
    else:
        log(f"✓ wiki imatrix present: {BASE_IMATRIX_WIKI.name}")

    # 2) hybrid_custom variant applied to the wiki imatrix.
    if not HYBRID_IMATRIX_WIKI.exists():
        with phase("build hybrid_custom imatrix from wiki E[a^2]"):
            imatrix.calibrate(
                variant="hybrid_custom",
                f16_gguf=F16_GGUF,
                base_imatrix=BASE_IMATRIX_WIKI,
                out_path=HYBRID_IMATRIX_WIKI,
            )
    else:
        log(f"✓ hybrid_wiki imatrix present: {HYBRID_IMATRIX_WIKI.name}")

    # 3) Three new quants.
    for qname, src_imatrix, qpath in [
        ("stock_custom", BASE_IMATRIX_CUSTOM, QUANT_STOCK_CUSTOM),
        ("stock_wiki",   BASE_IMATRIX_WIKI,   QUANT_STOCK_WIKI),
        ("hybrid_wiki",  HYBRID_IMATRIX_WIKI, QUANT_HYBRID_WIKI),
    ]:
        if qpath.exists():
            log(f"✓ {qpath.name} present")
            continue
        with phase(f"quantize Q4_K_M ({qname})"):
            gguf.quantize(F16_GGUF, qpath, "Q4_K_M",
                          imatrix=src_imatrix,
                          log=LOGS / f"quantize-{qname}.log")

    # 4) Bench the three new rows.
    n_params = bpw_mod.n_params(F16_GGUF)
    for label, quant in [
        ("stock/Q4_K_M-custom",  QUANT_STOCK_CUSTOM),
        ("stock/Q4_K_M-wiki",    QUANT_STOCK_WIKI),
        ("hybrid/Q4_K_M-wiki",   QUANT_HYBRID_WIKI),
    ]:
        with phase(f"bench {label}"):
            row = runner.bench_one(
                quant, label,
                reference_n_params=n_params,
                eval_dataset=CORPUS_EVAL,
                eval_baseline=EVAL_BASELINE,
                eval_ctx=8192,
                log_dir=LOGS,
                suite="full",
            )
            runner.append_row(RESULTS, row)
            log(f"  bpw={row.bpw:.3f} ppl={row.ppl} mean_kld={row.mean_kld} "
                f"same_top_p={row.same_top_p} decode={row.decode_tok_s}")

    # 5) Rename the prior hybrid row so labels are symmetric.
    relabel_row(RESULTS, old="imatrix/Q4_K_M-hybrid", new="hybrid/Q4_K_M-custom")

    # 6) Re-aggregate leaderboard.
    with phase("aggregate leaderboard"):
        md = aggregate.aggregate(RESULTS, weights=(1.0, 2.0, 1.0))
        LEADERBOARD.write_text(md)
        log(f"  wrote {LEADERBOARD.name}")
        print()
        print(md)

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
