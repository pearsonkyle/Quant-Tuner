#!/usr/bin/env python
"""Fetch the external SFT corpora and write them in the ``sft.jsonl.gz`` schema.

One round of the staged curriculum per invocation:

    # round 1 — broad conversational grounding
    python scripts/build_external_sft.py ultrachat --out out/corpora/round1-ultrachat

    # round 2 — tools / agents / reasoning
    python scripts/build_external_sft.py distill --out out/corpora/round2-distill

    # round 3 is our own corpus and is built by scripts/build_universal_corpus.py

The output plugs straight into ``scripts/build_sft_qat_corpus.py`` (and therefore
``qat.corpus.build_sft_corpus``) with no changes, because it IS that schema.

Every build VERIFIES rather than assumes, in the same spirit as ``data.universal``:
each source is rendered through the student's real chat template and the build fails if
tool schemas do not survive as objects, if a source that should carry tool calls emits
none, or if a control token present in the text does not tokenize to exactly one id.
A corpus that silently lost its tool calls trains a model that silently cannot call tools.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_tuner.data import external_sft as ext  # noqa: E402
from quant_tuner.data import reasoning as _reasoning  # noqa: E402
from quant_tuner.qat.corpus import load_tokenizer  # noqa: E402

#: Distillation configs worth pulling, and what each contributes. ``sft_agent`` is
#: deliberately absent — it is byte-identical to ``sft_tools`` (ext.DISTILL_ALIASES).
DISTILL_CONFIGS = {
    "sft_tools": "distill-tools",
    "sft_science": "distill-science",
    "sft_reasoning": "distill-reasoning",
}
UPSTREAM_SPLITS = ("train", "validation", "test")


def fetch_config(repo: str, config: str, split: str, cache: Path) -> list[dict]:
    """Download every shard of one config/split and return its rows."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo, repo_type="dataset")
             if f.startswith(f"data/{config}/{split}-") and f.endswith(".parquet")]
    if not files:
        return []
    rows: list[dict] = []
    for f in sorted(files):
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=str(cache))
        rows.extend(pq.read_table(p).to_pylist())
    return rows


def fetch_ultrachat(split: str, cache: Path, limit: int | None) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files(ext.ULTRACHAT_REPO, repo_type="dataset")
                   if f.endswith(".parquet") and f"/{split}-" in f)
    rows: list[dict] = []
    for f in files:
        p = hf_hub_download(ext.ULTRACHAT_REPO, f, repo_type="dataset", local_dir=str(cache))
        rows.extend(pq.read_table(p).to_pylist())
        if limit and len(rows) >= limit:
            return rows[:limit]
    return rows


def verify(records: list[dict], tok, *, label: str, expect_tools: bool) -> dict:
    """Render a sample through the real chat template and assert it survived.

    Checks the three failures that are invisible in the JSON but fatal in training:
    a tool schema that renders as a string, a tool-call source that emits no call
    markers, and a control token that tokenizes to more than one id.
    """
    sample = records[: min(200, len(records))]
    n_tools = n_calls = n_think = 0
    leaked = 0
    for r in sample:
        text = tok.apply_chat_template(r["messages"], tools=r["tools"], tokenize=False)
        if r["tools"]:
            n_tools += 1
            if "parameters_json" in text:
                leaked += 1
        n_calls += text.count("<tool_call>")
        # NON-EMPTY only. The Qwen3 template emits a bare `<think></think>` on the final
        # assistant turn when no reasoning is supplied, so a raw `<think>` count reports
        # 200 reasoning blocks on ultrachat, which has none. Measured, not theoretical.
        n_think += _reasoning.count_reasoning_blocks(text, nonempty_only=True)

    problems = []
    if leaked:
        problems.append(f"{leaked}/{n_tools} rendered a tool schema as a STRING "
                        f"(parameters_json leaked) — normalize_tool did not run")
    if expect_tools and n_calls == 0:
        problems.append("source should carry tool calls but the render emitted none")

    # Control tokens must survive as single ids on both stacks.
    joined = "".join(
        tok.apply_chat_template(r["messages"], tools=r["tools"], tokenize=False)
        for r in sample[:20])
    for marker in ("<|im_start|>", "<|im_end|>"):
        if marker in joined and len(tok.encode(marker, add_special_tokens=False)) != 1:
            problems.append(f"{marker} does not tokenize to a single id")

    if problems:
        sys.exit(f"[external-sft] {label} FAILED verification:\n  - " + "\n  - ".join(problems))

    return {"sampled": len(sample), "with_tools": n_tools,
            "tool_call_markers": n_calls, "nonempty_think_blocks": n_think}


def write_jsonl(records: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(records: list[dict]) -> dict:
    by_src = collections.Counter(r["source"] for r in records)
    by_split = collections.Counter(r["split"] for r in records)
    return {
        "n_records": len(records),
        "by_source": dict(by_src),
        "by_split": dict(by_split),
        "n_chars": sum(r["n_chars"] for r in records),
        "rows_with_tools": sum(1 for r in records if r["tools"]),
        "total_tool_calls": sum(r["n_tool_calls"] for r in records),
        "total_reasoning_turns": sum(r["n_reasoning"] for r in records),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round", choices=["ultrachat", "distill"])
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--cache", type=Path, default=Path("out/external/hf-sft"))
    ap.add_argument("--configs", nargs="*", default=["sft_tools", "sft_science"],
                    help=f"distill configs; available {sorted(DISTILL_CONFIGS)}")
    ap.add_argument("--limit", type=int, help="cap rows (smoke tests)")
    ap.add_argument("--drop-benchmarks", action="store_true",
                    help="drop rows sourced from public benchmark sets (ARC/SciQ/...) — "
                         "use whenever the run is later graded on multiple-choice evals")
    args = ap.parse_args()

    tok = load_tokenizer()
    records: list[dict] = []

    if args.round == "ultrachat":
        rows = fetch_ultrachat(ext.ULTRACHAT_SPLIT, args.cache, args.limit)
        print(f"[external-sft] ultrachat {ext.ULTRACHAT_SPLIT}: {len(rows):,} rows")
        recs = list(ext.convert_ultrachat_rows(rows))
        print("  verify:", verify(recs, tok, label="ultrachat", expect_tools=False))
        records += recs
    else:
        for cfg in args.configs:
            if cfg in ext.DISTILL_ALIASES:
                print(f"[external-sft] skipping {cfg}: byte-identical to "
                      f"{ext.DISTILL_ALIASES[cfg]}")
                continue
            src = DISTILL_CONFIGS.get(cfg, f"distill-{cfg}")
            got = 0
            for usplit in UPSTREAM_SPLITS:
                rows = fetch_config(ext.DISTILL_REPO, cfg, usplit, args.cache)
                if not rows:
                    continue
                if args.limit:
                    rows = rows[: args.limit]
                recs = list(ext.convert_distill_rows(
                    rows, source=src, split=usplit,
                    drop_benchmarks=args.drop_benchmarks))
                records += recs
                got += len(recs)
                print(f"[external-sft] {cfg}/{usplit}: {len(rows):,} rows -> "
                      f"{len(recs):,} records ({ext.UPSTREAM_SPLIT_MAP.get(usplit, usplit)})")
            if got:
                mine = [r for r in records if r["source"] == src]
                print(f"  verify {src}:",
                      verify(mine, tok, label=src, expect_tools=(cfg == "sft_tools")))

    if not records:
        sys.exit("[external-sft] no records built")

    out = args.out / "sft.jsonl.gz"
    write_jsonl(records, out)
    audit = summarize(records)
    (args.out / "external_sft_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"\n[external-sft] wrote {out}")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
