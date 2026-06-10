"""Build calibration / validation / eval text corpora.

Three corpora, one deterministic seed-42 session split for the logtrain
sources, and a separate external eval source so the reported PPL/KLD
numbers are not contaminated by anything the calibration loop saw.

  corpus.cal.txt   = ALL of wiki.test.raw + ~500k logtrain TRAIN-slice tokens
  corpus.val.txt   = ~10k logtrain TEST-slice tokens  +  calibration_supplement.txt
  corpus.eval.txt  = ~30k tokens each from external `eaddario/imatrix-calibration`
                     {code_small, math_small, tools_small} parquet files

Rationale: previously eval was wiki.test + a slice of logtrain — but both
appear in the calibration corpus, so PPL on that mix was partially measuring
fit to the calibration distribution rather than generalization. External
code/math/tools text gives the eval its own distribution that the search
never sees. The validation slice still pairs in-domain logtrain with the
under-represented `calibration_supplement.txt` so that cv-scored α
candidates are penalized when they overfit to logtrain's content mix.

The holdout slice (10%) of logtrain is unused here — it remains available
for tool-call eval sessions via the existing pipeline.

Usage:
  uv run python scripts/build_corpora.py \\
      --out out/corpora/<model-slug> \\
      --model out/<run>/model_extracted \\
      --wiki out/<run>/wiki.test.raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data import ingest, split

LOGTRAIN = REPO / "logtrain.jsonl"
SUPPLEMENT = REPO / "calibration_supplement.txt"

EAD_REPO = "eaddario/imatrix-calibration"
EVAL_DOMAINS = ("code_small", "math_small", "tools_small")
EAD_CACHE = REPO / "out" / "external" / "imatrix-calibration"


def _append(target: Path, addition: str) -> None:
    with open(target, "a") as f:
        if not addition.startswith("\n"):
            f.write("\n\n")
        f.write(addition)
        if not addition.endswith("\n"):
            f.write("\n")


def _download_parquet(domain: str) -> Path:
    """Return a local path to <domain>.parquet, downloading if absent."""
    from huggingface_hub import hf_hub_download

    EAD_CACHE.mkdir(parents=True, exist_ok=True)
    local = EAD_CACHE / f"{domain}.parquet"
    if local.exists():
        return local
    print(f"  downloading {EAD_REPO}/{domain}.parquet ...", file=sys.stderr)
    fetched = hf_hub_download(
        repo_id=EAD_REPO,
        filename=f"{domain}.parquet",
        repo_type="dataset",
        local_dir=EAD_CACHE,
    )
    return Path(fetched)


def _sample_parquet_text(
    parquet_path: Path, tok, target_tokens: int, seed: int,
) -> tuple[str, int, int]:
    """Sample `content`-column text until we hit `target_tokens` under `tok`.

    The eaddario parquet files often pack all content into a single very
    large row, so we additionally truncate at a token offset (deterministic
    via `seed`) when one row would exceed the target.

    Returns (joined_text, actual_token_count, n_rows_or_chunks_used).
    """
    import random

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["content"])
    rows = [r for r in table.column("content").to_pylist() if isinstance(r, str)]
    rng = random.Random(seed)
    rng.shuffle(rows)

    out_texts: list[str] = []
    total = 0
    used = 0
    for r in rows:
        if total >= target_tokens:
            break
        ids = tok(r, add_special_tokens=False)["input_ids"]
        remaining = target_tokens - total
        if len(ids) <= remaining:
            out_texts.append(r.strip())
            total += len(ids)
            used += 1
        else:
            # Random offset window so we don't always sample the head of huge rows.
            max_start = max(0, len(ids) - remaining)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            chunk = tok.decode(ids[start : start + remaining],
                               skip_special_tokens=True)
            out_texts.append(chunk.strip())
            total += remaining
            used += 1
            break
    return "\n\n".join(out_texts), total, used


def build(
    out_dir: Path,
    model_dir: Path,
    wiki_test: Path,
    cal_tokens: int,
    val_tokens: int,
    eval_tokens_per_domain: int,
    seed: int,
    val_supplement_override: Path | None = None,
) -> None:
    from transformers import AutoTokenizer

    assert LOGTRAIN.exists(), f"missing {LOGTRAIN}"
    wiki_test = wiki_test.resolve()
    out_dir = out_dir.resolve()
    assert wiki_test.exists(), f"missing {wiki_test}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)

    supplement_path: Path | None
    if val_supplement_override is not None:
        supplement_path = val_supplement_override.resolve()
        assert supplement_path.exists(), f"missing override supplement: {supplement_path}"
    else:
        supplement_path = SUPPLEMENT if SUPPLEMENT.exists() else None

    sessions = ingest.load_sessions(LOGTRAIN)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=seed,
    )
    print(
        f"sessions: train={len(splits['train'])}  "
        f"test={len(splits['test'])}  holdout={len(splits['holdout'])} (unused)",
        file=sys.stderr,
    )

    wiki_text = wiki_test.read_text()
    audit: dict = {
        "seed": seed,
        "logtrain_splits": {k: len(v) for k, v in splits.items()},
        "wiki_bytes": wiki_test.stat().st_size,
        "supplement": (
            str(supplement_path.relative_to(REPO))
            if supplement_path is not None and supplement_path.is_relative_to(REPO)
            else (str(supplement_path) if supplement_path is not None else None)
        ),
        "external_eval_repo": EAD_REPO,
        "external_eval_domains": list(EVAL_DOMAINS),
    }

    # --- 1) Calibration: ALL of wiki + ~500k logtrain (train slice) ----------
    cal_chunks, _kept, cal_total, cal_pack = split.stratified_pack(
        splits["train"], tok,
        target_tokens=cal_tokens, per_session_cap=6_000, seed=seed,
    )
    cal_logtrain = out_dir / "corpus.cal.logtrain.txt"
    split.write_corpus(cal_chunks, cal_logtrain)
    cal_corpus = out_dir / "corpus.cal.txt"
    sep = "" if wiki_text.endswith("\n") else "\n"
    cal_corpus.write_text(wiki_text + sep + "\n" + cal_logtrain.read_text())
    audit["calibration"] = {
        "path": str(cal_corpus.relative_to(REPO)),
        "target_logtrain_tokens": cal_tokens,
        "actual_logtrain_tokens": cal_total,
        "n_logtrain_sessions": len(cal_chunks),
        "pack_audit": cal_pack,
    }
    print(
        f"  calibration: {cal_corpus.relative_to(REPO)}  "
        f"(wiki + {cal_total:,} logtrain tokens)",
        file=sys.stderr,
    )

    # --- 2) Validation: logtrain (test slice) + supplement -------------------
    # When val_tokens == 0, skip logtrain entirely and use only the supplement
    # (the override path). This is how exp-020 builds a pure-MMMU val slice.
    val_corpus = out_dir / "corpus.val.txt"
    if val_tokens > 0:
        val_chunks, _kept2, val_total, val_pack = split.stratified_pack(
            splits["test"], tok,
            target_tokens=val_tokens, per_session_cap=3_000, seed=seed,
        )
        split.write_corpus(
            val_chunks, val_corpus,
            supplement=supplement_path,
        )
        audit["validation"] = {
            "path": str(val_corpus.relative_to(REPO)),
            "target_logtrain_tokens": val_tokens,
            "actual_logtrain_tokens": val_total,
            "n_logtrain_sessions": len(val_chunks),
            "pack_audit": val_pack,
            "supplement_source": (
                str(supplement_path) if supplement_path is not None else None
            ),
        }
        print(
            f"  validation:  {val_corpus.relative_to(REPO)}  "
            f"({val_total:,} logtrain tokens + supplement)",
            file=sys.stderr,
        )
    else:
        assert supplement_path is not None, (
            "val_tokens=0 requires val_supplement_override (or SUPPLEMENT) to "
            "provide the validation corpus"
        )
        val_corpus.write_bytes(supplement_path.read_bytes())
        audit["validation"] = {
            "path": str(val_corpus.relative_to(REPO)),
            "target_logtrain_tokens": 0,
            "actual_logtrain_tokens": 0,
            "n_logtrain_sessions": 0,
            "supplement_source": str(supplement_path),
            "source_bytes": supplement_path.stat().st_size,
        }
        print(
            f"  validation:  {val_corpus.relative_to(REPO)}  "
            f"(supplement-only: {supplement_path})",
            file=sys.stderr,
        )

    # --- 3) Eval: external code/math/tools, ~30k tokens each ----------------
    eval_corpus = out_dir / "corpus.eval.txt"
    eval_corpus.write_text("")
    eval_audit: dict = {}
    for i, domain in enumerate(EVAL_DOMAINS):
        pq_path = _download_parquet(domain)
        # Per-domain seed so reproducibility doesn't cross domains.
        text, ntok, nrows = _sample_parquet_text(
            pq_path, tok,
            target_tokens=eval_tokens_per_domain,
            seed=seed + i,
        )
        per_domain = out_dir / f"corpus.eval.{domain}.txt"
        per_domain.write_text(text)
        _append(eval_corpus, text)
        eval_audit[domain] = {
            "path": str(per_domain.relative_to(REPO)),
            "target_tokens": eval_tokens_per_domain,
            "actual_tokens": ntok,
            "n_rows": nrows,
            "source_parquet": str(pq_path.relative_to(REPO)),
        }
        print(
            f"  eval[{domain}]: {per_domain.relative_to(REPO)}  "
            f"({ntok:,} tokens from {nrows} rows)",
            file=sys.stderr,
        )
    audit["eval"] = {
        "path": str(eval_corpus.relative_to(REPO)),
        "target_tokens_per_domain": eval_tokens_per_domain,
        "domains": eval_audit,
    }
    eval_total_tokens = sum(d["actual_tokens"] for d in eval_audit.values())
    print(
        f"  eval:        {eval_corpus.relative_to(REPO)}  "
        f"({eval_total_tokens:,} tokens, {len(EVAL_DOMAINS)} domains)",
        file=sys.stderr,
    )

    # --- Disjointness on logtrain ------------------------------------------
    fp = ingest.session_fingerprint
    train_fp = {fp(s) for s in splits["train"]}
    test_fp = {fp(s) for s in splits["test"]}
    hold_fp = {fp(s) for s in splits["holdout"]}
    assert not (train_fp & test_fp), "train ∩ test non-empty"
    assert not (train_fp & hold_fp), "train ∩ holdout non-empty"
    assert not (test_fp & hold_fp), "test ∩ holdout non-empty"
    print("  disjointness: OK (logtrain train/test/holdout share no sessions)",
          file=sys.stderr)

    (out_dir / "corpora_audit.json").write_text(
        json.dumps(audit, indent=2, default=str)
    )
    print(f"  audit:       {(out_dir / 'corpora_audit.json').relative_to(REPO)}",
          file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", type=Path, required=True,
                   help="output directory for corpus.{cal,val,eval}.txt")
    p.add_argument("--model", type=Path, required=True,
                   help="HF model dir with tokenizer (e.g. out/<run>/model_extracted)")
    p.add_argument("--wiki", type=Path, required=True,
                   help="path to wiki.test.raw")
    p.add_argument("--cal-tokens", type=int, default=500_000,
                   help="target logtrain tokens in calibration corpus (default 500k)")
    p.add_argument("--val-tokens", type=int, default=10_000,
                   help="target logtrain tokens in validation corpus (default 10k)")
    p.add_argument("--eval-tokens-per-domain", type=int, default=30_000,
                   help="target tokens per external eval domain (default 30k)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-supplement", type=Path, default=None,
                   help="override the val supplement path (e.g. an MMMU file); "
                        "combine with --val-tokens 0 for a supplement-only val "
                        "corpus")
    a = p.parse_args()
    build(
        out_dir=a.out,
        model_dir=a.model,
        wiki_test=a.wiki,
        cal_tokens=a.cal_tokens,
        val_tokens=a.val_tokens,
        eval_tokens_per_domain=a.eval_tokens_per_domain,
        seed=a.seed,
        val_supplement_override=a.val_supplement,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
