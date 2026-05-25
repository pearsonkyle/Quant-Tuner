"""Add two extra IQ3_S variants for qwen — both use the *stock* (non-reweighted)
imatrix produced directly by `llama-imatrix`. They isolate the contribution
of (a) the calibration corpus and (b) the hybrid_custom re-weighting.

  * `IQ3_S-stock`  — stock imatrix from `corpus.train.txt` (the project's
    tool-call calibration corpus). Same E[a²] signal as `imatrix-custom.gguf`
    but used as-is (no hybrid_custom max combine).
  * `IQ3_S-wiki`   — stock imatrix calibrated on wikitext-2-raw-v1. Mirrors
    the standard llama.cpp recipe (E[a²] on a generic text corpus).

For each variant we:
  1. (build the imatrix if missing)
  2. quantize qwen f16 → IQ3_S using that imatrix
  3. run BPW + PPL + KLD + speed bench
  4. append a row to `qwen/results.csv` and `comparison.csv`

Usage::

    uv run python scripts/add_qwen_imatrix_variants.py
    uv run python scripts/add_qwen_imatrix_variants.py --skip-wiki     # just the stock-custom row
    uv run python scripts/add_qwen_imatrix_variants.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import runner as bench_runner   # noqa: E402
from quant_tuner.experiments import log, phase, step   # noqa: E402
from quant_tuner.models import llama_cpp                # noqa: E402
from quant_tuner.quantize import gguf as gguf_mod       # noqa: E402

WORKSPACE = REPO / "out/benchmark_9b_iq3s"
SLUG = "qwen"

# Match the existing imatrix corpus size so the comparison is apples-to-apples.
# corpus.train.txt is ~300 K tokens; we cap the wiki corpus similarly.
WIKI_MAX_BYTES = 2_500_000   # ≈ 600 K wikitext characters ≈ 200–300 K tokens


def _build_wiki_corpus(out_path: Path) -> None:
    """Concatenate wikitext-2-raw-v1 train rows into a single .txt file."""
    from datasets import load_dataset
    log(f"  downloading wikitext-2-raw-v1 …")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w") as f:
        for row in ds:
            t = row["text"]
            if not t.strip():
                continue
            f.write(t.rstrip("\n") + "\n")
            total += len(t)
            if total >= WIKI_MAX_BYTES:
                break
    log(f"  wrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


def _bench_and_record(label: str, quant_path: Path, f16: Path, eval_dataset: Path,
                      eval_baseline: Path, ref_n_params: int, log_dir: Path,
                      results_csv: Path) -> None:
    with phase(f"[{SLUG}] bench {label}"):
        row = bench_runner.bench_one(
            quant_path=quant_path,
            label=label,
            reference_n_params=ref_n_params,
            eval_dataset=eval_dataset,
            eval_baseline=eval_baseline,
            log_dir=log_dir,
            suite="full",
            bench_repetitions=10,
        )
        bench_runner.append_row(results_csv, row)
        log(f"  PPL={row.ppl:.4f}  KLD={row.mean_kld:.4f}  Same-Top-p={row.same_top_p:.2f}%  BPW={row.bpw:.3f}")


def _refresh_comparison_csv(workspace: Path) -> None:
    """Stitch per-slug results.csv into the top-level comparison.csv."""
    import csv as _csv
    out = workspace / "comparison.csv"
    rows: list[dict] = []
    fieldnames: list[str] = []
    for slug in ("qwen", "jackrong", "tesslate"):
        per_csv = workspace / slug / "results.csv"
        if not per_csv.exists():
            continue
        with per_csv.open() as f:
            reader = _csv.DictReader(f)
            for r in reader:
                r2 = {"source_model": slug, "hf_repo": "", **r}
                rows.append(r2)
                for k in r2:
                    if k not in fieldnames:
                        fieldnames.append(k)
    if not rows:
        return
    with out.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"  refreshed {out}  ({len(rows)} rows)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-stock", action="store_true",
                   help="skip the stock-custom variant")
    p.add_argument("--skip-wiki",  action="store_true",
                   help="skip the wiki variant")
    p.add_argument("--dry-run",    action="store_true")
    args = p.parse_args()

    base       = WORKSPACE / SLUG
    f16        = base / "model-f16.gguf"
    eval_ds    = base / "corpus.eval.txt"
    eval_kld   = base / "baseline.kld"
    imatrix_c  = base / "imatrix-custom.gguf"
    wiki_txt   = base / "corpus.wiki.txt"
    imatrix_w  = base / "imatrix-wiki.gguf"
    q_stock    = base / "IQ3_S-stock.gguf"
    q_wiki     = base / "IQ3_S-wiki.gguf"
    results    = base / "results.csv"
    log_dir    = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for need in (f16, eval_ds, eval_kld, imatrix_c):
        if not need.exists():
            log(f"ERROR: prerequisite missing: {need}")
            return 1

    # Reference param count from f16 (drives BPW). Reuses the project helper
    # so we don't hard-code the gguf import.
    from quant_tuner.bench.bpw import n_params as _n_params
    ref_n_params = _n_params(f16)
    log(f"reference n_params (qwen f16): {ref_n_params:,}")

    log(f"=== qwen extra IQ3_S variants ===")
    log(f"  f16:       {f16}")
    log(f"  eval:      {eval_ds}")
    log(f"  baseline:  {eval_kld}")
    log(f"  imatrix_c: {imatrix_c}")
    log(f"  out:       {q_stock.name}, {q_wiki.name}")

    if args.dry_run:
        log("dry-run — exiting")
        return 0

    if not args.skip_stock:
        # IQ3_S-stock — quantize with the existing base imatrix (no hybrid reweight)
        step(f"[{SLUG}] quantize IQ3_S-stock", q_stock,
             lambda: gguf_mod.quantize(
                 f16, q_stock, "IQ3_S",
                 imatrix=imatrix_c,
                 log=log_dir / "quantize_stock.log"))
        _bench_and_record("IQ3_S-stock", q_stock, f16, eval_ds, eval_kld,
                          ref_n_params, log_dir, results)

    if not args.skip_wiki:
        # imatrix-wiki — calibrate on wikitext
        step(f"[{SLUG}] wiki corpus", wiki_txt, lambda: _build_wiki_corpus(wiki_txt))
        step(f"[{SLUG}] llama-imatrix (wiki)", imatrix_w,
             lambda: llama_cpp.imatrix(
                 f16, wiki_txt, imatrix_w,
                 ctx=512, log=log_dir / "imatrix-wiki.log"))
        step(f"[{SLUG}] quantize IQ3_S-wiki", q_wiki,
             lambda: gguf_mod.quantize(
                 f16, q_wiki, "IQ3_S",
                 imatrix=imatrix_w,
                 log=log_dir / "quantize_wiki.log"))
        _bench_and_record("IQ3_S-wiki", q_wiki, f16, eval_ds, eval_kld,
                          ref_n_params, log_dir, results)

    _refresh_comparison_csv(WORKSPACE)
    log("=== DONE ===")
    log("Next: paste the new rows into EXPERIMENT.md or rerun the figure scripts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
