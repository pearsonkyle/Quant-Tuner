"""Benchmark Jackrong/Qwopus3.5-9B-Coder, Tesslate/OmniCoder-9B, and Qwen/Qwen3.5-9B.

For each model, produces:

  * F16 GGUF (with MTP nextn-predict layers bundled)
  * IQ3_S quantized with hybrid_custom imatrix from the logtrain corpus
  * KLD + speed bench for both variants (fp16 + IQ3_S)

After all models, merges per-model results.csv files into a single comparison.csv.

IQ3_S without calibration catastrophically degrades perplexity (~3.6M PPL measured).
The hybrid_custom imatrix variant (max of E[a²] and ||W[:,c]||² · E[a²] per tensor)
is required to make this quant level usable.

Workspace layout::

    out/benchmark_9b_iq3s/
        qwen/        → Qwen/Qwen3.5-9B
        jackrong/    → Jackrong/Qwopus3.5-9B-Coder
        tesslate/    → Tesslate/OmniCoder-9B
        comparison.csv

Usage::

    # Dry run
    uv run python scripts/benchmark_9b_models.py --logs logtrain.jsonl --dry-run

    # Single model to validate pipeline
    uv run python scripts/benchmark_9b_models.py --logs logtrain.jsonl --models qwen

    # Full three-model run
    uv run python scripts/benchmark_9b_models.py --logs logtrain.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.bench import bpw as bpw_mod
from quant_tuner.bench import kld, runner, speed as speed_mod
from quant_tuner.calibrate import imatrix
from quant_tuner.data import ingest, split
from quant_tuner.experiments import log, phase, step
from quant_tuner.models import extract, llama_cpp
from quant_tuner.paths import LLAMA_CPP_DIR
from quant_tuner.quantize import convert, gguf


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "qwen":     "Qwen/Qwen3.5-9B",
    "jackrong": "Jackrong/Qwopus3.5-9B-Coder",
    "tesslate": "Tesslate/OmniCoder-9B",
}


# ---------------------------------------------------------------------------
# Per-model workspace layout
# ---------------------------------------------------------------------------

class Layout:
    def __init__(self, work: Path) -> None:
        self.work = work
        self.logs = work / "logs"

        self.model_dir = work / "model_extracted"
        self.f16 = work / "model-f16.gguf"

        self.corpus_train = work / "corpus.train.txt"
        self.corpus_eval = work / "corpus.eval.txt"

        self.imatrix_custom = work / "imatrix-custom.gguf"
        self.imatrix_hybrid_custom = work / "imatrix-hybrid_custom.gguf"

        self.q_custom = work / "IQ3_S-custom.gguf"

        self.kld_baseline = work / "baseline.kld"
        self.results = work / "results.csv"

    def all_quants(self) -> list[tuple[str, Path]]:
        return [
            ("IQ3_S-custom", self.q_custom),
        ]


# ---------------------------------------------------------------------------
# MTP verification
# ---------------------------------------------------------------------------

def _check_mtp(f16: Path) -> None:
    """Read GGUF metadata and report the nextn_predict_layers count."""
    p = str(LLAMA_CPP_DIR / "gguf-py")
    if p not in sys.path:
        sys.path.insert(0, p)
    from gguf import GGUFReader

    reader = GGUFReader(str(f16))
    kv = {field.name: field for field in reader.fields.values()}

    nextn = 0
    for key in kv:
        if "nextn_predict_layers" in key or "num_nextn_predict" in key:
            val = kv[key]
            try:
                nextn = int(val.parts[-1].flat[0])
            except Exception:
                pass
            break

    if nextn > 0:
        log(f"  MTP: num_nextn_predict_layers = {nextn} ✓")
    else:
        log(f"  WARNING: num_nextn_predict_layers not found or 0 in {f16.name}")
        log("  MTP speed-up will not be available for this GGUF.")


# ---------------------------------------------------------------------------
# MTP injection helpers
# ---------------------------------------------------------------------------

def _has_mtp_in_safetensors(model_dir: Path) -> bool:
    """Return True if model_dir's safetensors already contain mtp.* tensors."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        wmap: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        return any(k.startswith("mtp.") for k in wmap)
    # Single-file fallback
    st = model_dir / "model.safetensors"
    if st.exists():
        from safetensors import safe_open
        with safe_open(str(st), framework="pt") as f:
            return any(k.startswith("mtp.") for k in f.keys())
    return False


def _inject_mtp_weights(target_dir: Path, donor_dir: Path) -> None:
    """Copy mtp.* tensors from donor_dir into target_dir as a new safetensors shard."""
    from safetensors.torch import load_file, save_file

    donor_index_path = donor_dir / "model.safetensors.index.json"
    if not donor_index_path.exists():
        raise FileNotFoundError(
            f"Donor model at {donor_dir} has no safetensors index"
        )

    donor_wmap: dict[str, str] = json.loads(donor_index_path.read_text())["weight_map"]
    mtp_by_shard: dict[str, list[str]] = {}
    for key, shard in donor_wmap.items():
        if key.startswith("mtp."):
            mtp_by_shard.setdefault(shard, []).append(key)

    if not mtp_by_shard:
        raise ValueError(f"Donor {donor_dir} has no mtp.* tensors to inject")

    # Load MTP tensors (may span multiple donor shards)
    mtp_tensors: dict[str, Any] = {}
    for shard, keys in mtp_by_shard.items():
        loaded = load_file(str(donor_dir / shard))
        mtp_tensors.update({k: loaded[k] for k in keys})

    # Write as a dedicated shard in target
    out_shard = "model-mtp.safetensors"
    save_file(mtp_tensors, str(target_dir / out_shard))
    log(f"  wrote {len(mtp_tensors)} MTP tensors -> {out_shard}")

    # Update target's safetensors index
    target_index_path = target_dir / "model.safetensors.index.json"
    if target_index_path.exists():
        target_index = json.loads(target_index_path.read_text())
        for k in mtp_tensors:
            target_index["weight_map"][k] = out_shard
        target_index_path.write_text(json.dumps(target_index, indent=2))
        log(f"  updated safetensors index with {len(mtp_tensors)} new entries")


def _ensure_mtp(
    slug: str,
    model_dir: Path,
    f16: Path,
    donor_dir: Path | None,
) -> None:
    """Ensure MTP weights are present; inject from donor or zero config if absent."""
    if _has_mtp_in_safetensors(model_dir):
        log(f"  [{slug}] MTP weights present in safetensors ✓")
        return

    config_path = model_dir / "config.json"
    config: dict[str, Any] = json.loads(config_path.read_text())
    n_mtp = config.get("mtp_num_hidden_layers", 0)

    if not n_mtp:
        log(f"  [{slug}] mtp_num_hidden_layers=0 — no MTP expected, skipping")
        return

    log(f"  [{slug}] config claims mtp_num_hidden_layers={n_mtp} "
        f"but no mtp.* tensors in safetensors")

    if donor_dir is not None and donor_dir.exists() and _has_mtp_in_safetensors(donor_dir):
        with phase(f"[{slug}] inject MTP weights from qwen donor"):
            _inject_mtp_weights(model_dir, donor_dir)
        if f16.exists():
            f16.unlink()
            log(f"  deleted stale {f16.name} — will reconvert with MTP weights")
    else:
        log(f"  [{slug}] no donor with MTP weights available")
        log(f"  [{slug}] zeroing mtp_num_hidden_layers in config to avoid broken GGUF")
        config["mtp_num_hidden_layers"] = 0
        config_path.write_text(json.dumps(config, indent=2))
        if f16.exists():
            f16.unlink()
            log(f"  deleted stale {f16.name} — will reconvert without MTP")


# ---------------------------------------------------------------------------
# Stage 1: corpus preparation (same logic as reproduce_table.py)
# ---------------------------------------------------------------------------

def _prepare_corpora(
    L: Layout,
    logtrain: Path,
    supplement: Path | None,
    train_max: int,
    eval_max: int,
) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(L.model_dir), fix_mistral_regex=True)
    sessions = ingest.load_sessions(logtrain)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    log(f"  {len(sessions)} sessions after filtering")

    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
    log(f"  split: train={len(splits['train'])} test={len(splits['test'])} "
        f"holdout={len(splits['holdout'])}")

    chunks, _k, total, audit = split.stratified_pack(
        splits["train"], tok, target_tokens=train_max,
        per_session_cap=6_000, seed=42,
    )
    split.write_corpus(chunks, L.corpus_train, supplement=supplement)
    (L.work / "train_audit.json").write_text(
        json.dumps(audit, indent=2, default=str)
    )
    extra = f" (+ supplement {supplement.name})" if supplement else ""
    log(f"  train corpus: {total:,} tokens -> {L.corpus_train.name}{extra}")

    ev_chunks, _ek, ev_total, ev_audit = split.stratified_pack(
        splits["test"], tok, target_tokens=eval_max,
        per_session_cap=4_000, seed=43,
    )
    split.write_corpus(ev_chunks, L.corpus_eval)
    (L.work / "eval_audit.json").write_text(
        json.dumps(ev_audit, indent=2, default=str)
    )
    log(f"  eval corpus:  {ev_total:,} tokens -> {L.corpus_eval.name}")


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------

def _run_model(
    slug: str,
    model: str,
    L: Layout,
    logtrain: Path,
    supplement: Path | None,
    train_max: int,
    eval_max: int,
    skip_speed_rebench: bool,
    mtp_donor_dir: Path | None = None,
) -> None:
    L.work.mkdir(parents=True, exist_ok=True)
    L.logs.mkdir(parents=True, exist_ok=True)

    # 1. Extract / download model
    step(f"[{slug}] extract / fetch HF model", L.model_dir / "config.json",
         lambda: extract.extract_text_lm(
             source=model, output_dir=L.model_dir, keep_mtp=True))

    # 2. Ensure MTP weights present; inject from donor or zero config if absent
    _ensure_mtp(slug, L.model_dir, L.f16, mtp_donor_dir)

    # 3. Convert to F16 GGUF
    step(f"[{slug}] convert HF -> F16 GGUF", L.f16,
         lambda: convert.hf_to_f16_gguf(
             L.model_dir, L.f16, log=L.logs / "convert.log"))

    # 4. Verify MTP layers present in GGUF
    _check_mtp(L.f16)

    # 5. Prepare calibration + eval corpora from logtrain
    step(f"[{slug}] prepare corpora", [L.corpus_train, L.corpus_eval],
         lambda: _prepare_corpora(L, logtrain, supplement, train_max, eval_max))

    # 6. Base imatrix (stock E[a²]) from logtrain corpus
    step(f"[{slug}] llama-imatrix (custom)", L.imatrix_custom,
         lambda: llama_cpp.imatrix(
             L.f16, L.corpus_train, L.imatrix_custom,
             ctx=512, log=L.logs / "imatrix-custom.log"))

    # 7. hybrid_custom reweighting
    step(f"[{slug}] hybrid_custom imatrix", L.imatrix_hybrid_custom,
         lambda: imatrix.calibrate(
             variant="hybrid_custom", f16_gguf=L.f16,
             base_imatrix=L.imatrix_custom, out_path=L.imatrix_hybrid_custom))

    # 8. Quantize to IQ3_S
    step(f"[{slug}] quantize IQ3_S (custom)", L.q_custom,
         lambda: gguf.quantize(
             L.f16, L.q_custom, "IQ3_S", imatrix=L.imatrix_hybrid_custom,
             log=L.logs / "quantize-IQ3_S-custom.log"))

    # 9. KLD baseline from F16
    step(f"[{slug}] build F16 KLD baseline", L.kld_baseline,
         lambda: kld.build_baseline(
             L.f16, L.corpus_eval, L.kld_baseline,
             ctx=8192, log=L.logs / "baseline.log"))

    # 10-11. Bench: fp16 + IQ3_S
    _stage_bench(slug, L)

    if not skip_speed_rebench:
        with phase(f"[{slug}] 10-rep speed rebench"):
            _stage_speed_rebench(L)


# ---------------------------------------------------------------------------
# Bench stage
# ---------------------------------------------------------------------------

def _stage_bench(slug: str, L: Layout) -> None:
    n_params = bpw_mod.n_params(L.f16)
    log(f"  n_params = {n_params:,}")

    benched: set[str] = set()
    if L.results.exists():
        with L.results.open() as f:
            benched = {r["model"] for r in csv.DictReader(f)}

    rows: list[tuple[str, Path]] = [("baseline/fp16", L.f16)]
    rows += [(label, q) for label, q in L.all_quants()]

    for label, quant in rows:
        if label in benched:
            log(f"  ✓ {label} already in results.csv — skipping")
            continue
        if not quant.exists():
            log(f"  [skip] {label}: missing {quant.name}")
            continue
        with phase(f"[{slug}] bench {label}"):
            row = runner.bench_one(
                quant, label,
                reference_n_params=n_params,
                eval_dataset=L.corpus_eval,
                eval_baseline=L.kld_baseline,
                eval_ctx=8192,
                log_dir=L.logs,
                suite="full",
                bench_repetitions=10,
            )
            runner.append_row(L.results, row)
            log(f"  bpw={row.bpw:.3f} mean_kld={row.mean_kld} "
                f"same_top_p={row.same_top_p} decode={row.decode_tok_s}")


def _stage_speed_rebench(L: Layout) -> None:
    assert L.results.exists(), "results.csv missing — bench stage must run first"
    with L.results.open() as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        by_label = {r["model"]: r for r in reader}
    for col in runner.CSV_COLUMNS:
        if col not in fields:
            fields.append(col)

    targets = [("baseline/fp16", L.f16)] + list(L.all_quants())
    for label, model in targets:
        if not model.exists() or label not in by_label:
            continue
        with phase(f"  speed rebench {label}"):
            m = speed_mod.evaluate(
                model, repetitions=10,
                log=L.logs / f"speed_{model.stem}.log",
            )
            row = by_label[label]
            for k in ("prefill_tok_s", "prefill_stdev",
                      "decode_tok_s", "decode_stdev",
                      "ttft_2k_ms", "ttft_stdev_ms"):
                v = getattr(m, k)
                row[k] = "" if v is None else v
            row["bench_repetitions"] = m.n_repetitions
            with L.results.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in by_label.values():
                    w.writerow({k: r.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _merge_results(
    root: Path,
    slugs: list[str],
    model_map: dict[str, str],
) -> Path:
    """Merge per-model results.csv files into a single comparison.csv."""
    out_path = root / "comparison.csv"
    all_rows: list[dict] = []
    all_fields: list[str] = ["source_model", "hf_repo"]

    for slug in slugs:
        csv_path = root / slug / "results.csv"
        if not csv_path.exists():
            log(f"  [skip] {slug}/results.csv not found")
            continue
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for col in (reader.fieldnames or []):
                if col not in all_fields:
                    all_fields.append(col)
            for row in reader:
                row["source_model"] = slug
                row["hf_repo"] = model_map.get(slug, "")
                all_rows.append(row)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, "") for k in all_fields})

    log(f"  wrote {out_path} ({len(all_rows)} rows)")
    return out_path


def _print_table(comparison_csv: Path) -> None:
    """Print a compact comparison table to stdout."""
    if not comparison_csv.exists():
        return
    rows = []
    with comparison_csv.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    cols = ["source_model", "model", "bpw", "mean_kld", "same_top_p",
            "decode_tok_s", "ppl"]
    header = f"{'model':12s}  {'variant':22s}  {'bpw':>5s}  {'mean_kld':>10s}  "
    header += f"{'same_top_p':>10s}  {'decode_tok/s':>12s}  {'ppl':>10s}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        src = r.get("source_model", "")[:12]
        variant = r.get("model", "")[:22]
        bpw = r.get("bpw", "")
        mkld = r.get("mean_kld", "")
        stp = r.get("same_top_p", "")
        dec = r.get("decode_tok_s", "")
        ppl = r.get("ppl", "")
        print(f"{src:12s}  {variant:22s}  {bpw:>5s}  {mkld:>10s}  "
              f"{stp:>10s}  {dec:>12s}  {ppl:>10s}")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--logs", type=Path, default=REPO / "logtrain.jsonl",
                   help="Usage-log JSONL (logtrain.jsonl)")
    p.add_argument("--workspace", type=Path, default=REPO / "out" / "benchmark_9b_iq3s",
                   help="Root output directory")
    p.add_argument("--supplement", type=Path,
                   default=REPO / "calibration_supplement.txt",
                   help="Optional supplement appended to train corpus")
    p.add_argument("--models", default="qwen,jackrong,tesslate",
                   help="Comma-separated subset of: qwen,jackrong,tesslate")
    p.add_argument("--train-max-tokens", type=int, default=250_000)
    p.add_argument("--eval-max-tokens", type=int, default=50_000)
    p.add_argument("--skip-speed-rebench", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without running anything")
    args = p.parse_args()

    requested = [m.strip() for m in args.models.split(",")]
    unknown = [m for m in requested if m not in MODELS]
    if unknown:
        p.error(f"Unknown model key(s): {unknown}. Choose from: {list(MODELS)}")

    supplement = (
        args.supplement
        if args.supplement and args.supplement.exists()
        else None
    )

    log("=" * 60)
    log("Benchmark: IQ3_S hybrid_custom × logtrain (custom corpus)")
    log(f"models:    {requested}")
    log(f"workspace: {args.workspace}")
    log(f"logtrain:  {args.logs}")
    log(f"supplement:{supplement or '(none)'}")
    log(f"budgets:   train={args.train_max_tokens:,}  eval={args.eval_max_tokens:,}")
    log("=" * 60)

    if args.dry_run:
        for slug in requested:
            log(f"  would process: {slug} → {MODELS[slug]}")
            log(f"                 workspace: {args.workspace / slug}")
        return 0

    t0 = time.time()
    # Qwen/Qwen3.5-9B is the canonical MTP donor: it has verified mtp.* weights
    # and is always processed first (or already extracted from a prior run).
    mtp_donor_dir = args.workspace / "qwen" / "model_extracted"

    for slug in requested:
        model = MODELS[slug]
        L = Layout(args.workspace / slug)
        log("")
        log(f"{'='*60}")
        log(f"  MODEL: {model} [{slug}]")
        log(f"{'='*60}")
        with phase(f"full pipeline for {slug}"):
            _run_model(
                slug=slug,
                model=model,
                L=L,
                logtrain=args.logs,
                supplement=supplement,
                train_max=args.train_max_tokens,
                eval_max=args.eval_max_tokens,
                skip_speed_rebench=args.skip_speed_rebench,
                # qwen uses its own native MTP weights; others borrow from it
                mtp_donor_dir=mtp_donor_dir if slug != "qwen" else None,
            )

    log("")
    with phase("merge comparison table"):
        comparison = _merge_results(args.workspace, requested, MODELS)
        _print_table(comparison)

    log(f"=== ALL DONE in {(time.time() - t0) / 3600:.2f} h ===")
    log(f"    results: {args.workspace}/comparison.csv")
    log("    MTP speed: uv run python scripts/bench_mtp_speed.py \\")
    log("               --model out/benchmark_9b_iq3s/<slug>/IQ3_S-custom.gguf \\")
    log("               --reps 5 --n-max 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
