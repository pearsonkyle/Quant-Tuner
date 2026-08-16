"""Build ONLY the external + general eval corpora for exp-060.

``universal.build`` cannot run here: 42% of the calibration corpus comes from
``datasets/agent-logs/data/*.jsonl.gz``, which is local-only and absent on this box
(the calibration corpus was transferred prebuilt instead). But the two *external* eval
corpora depend on nothing local — they are sampled from the ``eaddario/imatrix-calibration``
parquet files plus the model tokenizer.

This replicates section 7 of ``universal.build`` verbatim (same helper, same seeds, same
per-domain token budgets, same concatenation form) so the eval distributions — and hence
the PPL/KLD numbers — are identical to what a full build would have produced.

    PYTHONPATH=src .venv/bin/python scripts/exp060_build_eval_corpora.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data import external

# Mirrors UniversalConfig defaults — these must not drift from data/universal.py.
SEED = 42
EVAL_TOKENS_PER_DOMAIN = 30_000
GENERAL_EVAL_TOKENS = 30_000


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="exp-060")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    a = p.parse_args()

    out = REPO / "out" / a.run / "corpora"
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)

    audit: dict = {}

    # --- external: code_small + math_small + tools_small, concatenated -----------------
    ext_corpus = out / "corpus.eval.txt"
    ext_corpus.write_text("")
    domains: dict = {}
    for i, domain in enumerate(external.EVAL_DOMAINS):
        text, n, rows_used = external.sample_parquet_text(
            external.download_parquet(domain), tok,
            target_tokens=EVAL_TOKENS_PER_DOMAIN, seed=SEED + i,
        )
        (out / f"corpus.eval.{domain}.txt").write_text(text)
        with ext_corpus.open("a") as fh:
            fh.write(text.rstrip() + "\n\n")
        domains[domain] = {"tokens": n, "rows": rows_used}
        print(f"  {domain}: {n:,} tokens ({rows_used} rows)")
    audit["external"] = {
        "path": str(ext_corpus), "repo": external.EAD_REPO, "domains": domains,
        "used_for": "headline PPL/KLD (code/math/tools) — disjoint from every cal source",
    }

    # --- general: combined_en_tiny ----------------------------------------------------
    gtext, gn, grows = external.sample_parquet_text(
        external.download_parquet(external.GENERAL_EVAL_DOMAIN), tok,
        target_tokens=GENERAL_EVAL_TOKENS, seed=SEED + len(external.EVAL_DOMAINS),
    )
    (out / "corpus.eval.general.txt").write_text(gtext)
    audit["general"] = {
        "path": str(out / "corpus.eval.general.txt"),
        "domain": external.GENERAL_EVAL_DOMAIN, "tokens": gn, "rows": grows,
    }
    print(f"  {external.GENERAL_EVAL_DOMAIN}: {gn:,} tokens ({grows} rows)")

    (out / "eval_corpora_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"\nwrote {ext_corpus} ({ext_corpus.stat().st_size:,} bytes)")
    print(f"wrote {out / 'corpus.eval.general.txt'} "
          f"({(out / 'corpus.eval.general.txt').stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
