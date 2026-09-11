#!/usr/bin/env python3
"""Render a published calibration split into the corpus.cal.txt both quantizers eat.

GPTQ (`run_vllm_ptq.py --corpus`) and `llama-imatrix -f` both want a flat text
file. The published split is conversations -- `messages` as JSON strings plus a
`tools` blob -- so it has to go through the chat template first, using the repo's
own `template_session` rather than a reimplementation: calibration has to see the
token distribution the model actually serves, and for a tool-use model the tool
SCHEMAS are a large part of that. Rendering without `tools=` would drop every
schema and calibrate on a distribution the served model never sees.

Three things are checked rather than assumed, because each fails silently:

  tokens vs the split's own tier_meta
      The publisher recorded an exact token count. Re-rendering here should
      reproduce it. A large miss means the tokenizer or the chat template is not
      the one the split was built with, and the calibration would be measuring a
      different corpus than its metadata claims.

  tool-call markers present, PER SOURCE
      A tool-use model calibrated on a corpus with no tool-call spans is
      miscalibrated exactly where it matters. Checking only the total hides a
      source that quietly stopped carrying them, because the other sources keep
      the total non-zero.

  windows that hit the cap
      The split is packed at a fixed ctx. Feeding it to a quantizer at a
      DIFFERENT --ctx does not calibrate on longer trajectories -- it glues
      unrelated windows together, since the cut already happened at pack time.
      The audit records the pack ctx so the --ctx used downstream can be checked
      against it.

    python scripts/cal_corpus_from_parquet.py \
        --parquet-dir <snapshot>/data/calibration-15m-v65536-ctx32k \
        --out-dir out/e4b-v65536/cal-15m-ctx32k
"""
import argparse
import collections
import glob
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer",
                    default="/workspace/models/gemma4-e4b-stage0-32k-v65536/final")
    ap.add_argument("--token-tolerance", type=float, default=0.02,
                    help="max fractional deviation from the split's recorded token count")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from quant_tuner.data.split import template_session, write_corpus
    from quant_tuner.data.universal import _scan_tool_calls

    src_dir = Path(a.parquet_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {}
    mp = src_dir / "tier_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
        print(f"split: {meta.get('label')}  "
              f"recorded {meta.get('tokens'):,} tokens over {meta.get('n'):,} rows "
              f"at ctx {meta.get('max_tokens'):,}")

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    shards = sorted(glob.glob(str(src_dir / "*.parquet")))
    if not shards:
        raise SystemExit(f"no parquet shards under {src_dir}")
    print(f"rendering {len(shards)} shards through the chat template")

    chunks, sources, ntoks = [], [], []
    for i, sh in enumerate(shards):
        t = pq.read_table(sh).to_pylist()
        for row in t:
            messages = [json.loads(m) for m in row["messages"]]
            tools = json.loads(row["tools"]) if row.get("tools") else None
            text, n = template_session(tok, messages, tools)
            chunks.append(text)
            sources.append(row.get("source", "?"))
            ntoks.append(n)
        print(f"  shard {i+1}/{len(shards)}: {len(chunks):,} rows, "
              f"{sum(ntoks):,} tokens", flush=True)

    cal = out / "corpus.cal.txt"
    write_corpus(chunks, cal)
    with gzip.open(out / "corpus.cal.jsonl.gz", "wt") as fh:
        for i, (c, s, n) in enumerate(zip(chunks, sources, ntoks)):
            fh.write(json.dumps({"i": i, "source": s, "n_chars": len(c),
                                 "n_tokens": n, "text": c}) + "\n")

    total = sum(ntoks)
    per_source = collections.Counter()
    for s, n in zip(sources, ntoks):
        per_source[s] += n
    cap = meta.get("max_tokens")
    at_cap = sum(1 for n in ntoks if cap and n >= cap - 8)

    text = cal.read_text()
    scan = _scan_tool_calls(text)
    by_source = {}
    for s in sorted(set(sources)):
        joined = "\n\n".join(c for c, ss in zip(chunks, sources) if ss == s)
        by_source[s] = _scan_tool_calls(joined)["tool_call_marker_total"]

    audit = {
        "source_split": str(src_dir), "tier_meta": meta,
        "tokenizer": a.tokenizer, "pack_ctx": cap,
        "rows": len(chunks), "tokens_rendered": total,
        "tokens_recorded": meta.get("tokens"),
        "bytes": cal.stat().st_size,
        "rows_at_pack_cap": at_cap,
        "per_source_tokens": dict(per_source.most_common()),
        "tool_calls": scan,
        "tool_call_markers_by_source": by_source,
    }
    (out / "cal_audit.json").write_text(json.dumps(audit, indent=1))

    print(f"\nwrote {cal}  ({cal.stat().st_size/1e6:.1f} MB)")
    print(f"rows {len(chunks):,}   tokens {total:,}   at pack cap {at_cap:,}")
    print(f"tool-call markers {scan['tool_call_marker_total']:,}   "
          f"tool-response markers {scan['tool_response_marker_total']:,}")

    rec = meta.get("tokens")
    if rec:
        dev = abs(total - rec) / rec
        print(f"vs recorded {rec:,}: {total - rec:+,} ({dev*100:.2f}%)")
        if dev > a.token_tolerance:
            raise SystemExit(
                f"rendered token count deviates {dev*100:.1f}% from the split's own "
                f"metadata (tolerance {a.token_tolerance*100:.0f}%). The tokenizer or "
                f"chat template is probably not the one this split was built with; "
                f"calibrating now would use a corpus its metadata does not describe.")

    if scan["tool_call_marker_total"] == 0:
        raise SystemExit(
            "no tool-call markers in the rendered corpus. Either `tools` was not "
            "passed to the template or the template does not emit them -- either "
            "way this corpus would miscalibrate a tool-use model.")
    empty = [s for s, n in by_source.items() if n == 0]
    if empty:
        print(f"\nNOTE: sources with no tool-call markers: {', '.join(empty)}")
        print("      expected for prose sources; investigate if an agentic one is listed.")
    print("\nCAL CORPUS DONE")


if __name__ == "__main__":
    raise SystemExit(main())
