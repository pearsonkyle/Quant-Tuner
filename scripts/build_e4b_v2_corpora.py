"""Calibration + eval corpora for the pruned-vocabulary gemma-4-E4B runs.

Sibling of ``build_universal_corpus.py``, which reads ``datasets/`` in this repo.
This one reads the LLM-Training-Kit v2 SFT pack instead -- the same file the
model was fine-tuned on -- because a quant should be calibrated on the
distribution it will actually serve.

Two rules this inherits from build_universal_corpus, and neither is optional:

  * **Calibration draws only from the TRAIN slice.** The trainer's split is
    reproduced exactly (two seeded ``train_test_split`` calls, seed 42, same
    JSONL order), so the test holdout stays clean. That holdout is also where
    the agentic benchmark sessions come from, so calibrating on it would score
    the quant on its own calibration set.
  * **Each eval corpus is its own distribution.** They get separate files and
    separate ``kld.build_baseline`` runs; never concatenate them.

Stratification: the pack is 34 sources and badly unbalanced (mtg-instruct alone
is 19.2%). Proportional sampling would spend a fifth of the calibration budget
on one card game. Sources are round-robined and capped at ``--max-source-share``
so the budget buys domain coverage rather than the mode of the corpus.

Usage:
    uv run python scripts/build_e4b_v2_corpora.py \\
        --pack /workspace/sft-corpus-v2-rw-v65536/sft-131072.jsonl \\
        --model /workspace/models/gemma4-e4b-qat-v65536-text \\
        --out out/e4b-v2-corpora --budget-tokens 5_000_000 --ctx 32768
"""

from __future__ import annotations

import argparse
import collections
import math
import json
import random
import sys
from pathlib import Path


def replay_split(
    n: int,
    seed: int = 42,
    train_ratio: float = 0.98,
    eval_ratio: float = 0.01,
    test_ratio: float = 0.01,
) -> tuple[set[int], set[int], set[int]]:
    """Reproduce the trainer's split exactly, as index sets.

    llmtk (`trainer.py`) does two seeded `train_test_split` calls on the combined
    dataset. Neither is stratified, so the partition depends only on the ROW
    COUNT and the seed -- not on row content. That lets us split a throwaway
    index-only Dataset of the same length and get provably the same membership,
    without materializing the 5.1 GiB messages column (whose `list<string>`
    child offsets would blow Arrow's 2 GiB cap the moment `train_test_split`
    pickles the table to fingerprint it).

    Do NOT reimplement the shuffle by hand: matching `datasets`' generator is
    the whole point, and getting it subtly wrong leaks test rows into the
    calibration corpus with nothing to announce it.
    """
    from datasets import Dataset

    ds = Dataset.from_dict({"i": list(range(n))})
    # Mirror trainer.py's own arithmetic exactly. Note the first split holds out
    # eval+test together, so `test` ends up ~2% even though test_ratio is 0.01.
    s1 = ds.train_test_split(test_size=eval_ratio + test_ratio, seed=seed)
    test = set(s1["test"]["i"])
    eval_in_train = eval_ratio / (train_ratio + eval_ratio)
    s2 = s1["train"].train_test_split(test_size=eval_in_train, seed=seed)
    ev = set(s2["test"]["i"])
    train = set(s2["train"]["i"])
    assert len(train) + len(ev) + len(test) == n
    assert not (train & test) and not (train & ev) and not (ev & test)
    return train, ev, test


def _decode(v):
    """The pack stores messages as a list of JSON STRINGS and tools as one JSON
    string -- Arrow `list<large_string>`, which is what lets llmtk carry a 5 GiB
    messages column past the int32 offset cap. Feeding those to
    `apply_chat_template` raises; they have to be parsed back first.
    """
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, str):
                try:
                    out.append(json.loads(item))
                except json.JSONDecodeError:
                    return None
            elif isinstance(item, dict):
                out.append(item)
        return out
    return v


def render(tok, row: dict, failures: collections.Counter) -> str | None:
    msgs = _decode(row.get("messages"))
    tools = _decode(row.get("tools"))
    if not msgs:
        failures["no_messages"] += 1
        return None
    try:
        return tok.apply_chat_template(msgs, tools=tools or None, tokenize=False)
    except Exception as e:
        failures[f"template:{type(e).__name__}"] += 1
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="sft-131072.jsonl from LLM-Training-Kit")
    ap.add_argument("--model", required=True, help="dir holding the restricted tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-tokens", type=int, default=5_000_000)
    ap.add_argument("--ctx", type=int, default=32768,
                    help="window the PTQ/imatrix pass will chunk at; only used to warn "
                         "when the budget cannot fill enough windows")
    ap.add_argument("--max-source-share", type=float, default=0.08,
                    help="cap any single source at this fraction of the budget")
    ap.add_argument("--eval-tokens", type=int, default=400_000,
                    help="budget for the held-out general eval corpus")
    ap.add_argument("--eval-agentic-tokens", type=int, default=2_000_000,
                    help="budget for the held-out agentic eval corpus. Larger on "
                         "purpose: agent sessions run ~50k tokens each, so the "
                         "general slice's budget buys only a handful of them and "
                         "the 'distribution' ends up being 8 conversations.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emit-records", action="store_true",
                    help="also write the selection as parquet shards, shaped like "
                         "the pack's published tiers, for upload as a dataset split")
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--label", default="calibration-15m-v65536")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"reading {args.pack} ...", flush=True)
    rows: list[dict] = []
    with open(args.pack) as f:
        for line in f:
            rows.append(json.loads(line))
    n = len(rows)
    train_i, eval_i, test_i = replay_split(n, args.seed)
    print(f"  {n:,} rows -> train {len(train_i):,} / eval {len(eval_i):,} / test {len(test_i):,}",
          flush=True)

    by_source: dict[str, list[int]] = collections.defaultdict(list)
    for i in train_i:
        by_source[rows[i].get("source") or "(none)"].append(i)
    failures: collections.Counter = collections.Counter()
    rng = random.Random(args.seed)
    for v in by_source.values():
        rng.shuffle(v)

    cap = int(args.budget_tokens * args.max_source_share)
    order = sorted(by_source)
    cursors = {s: 0 for s in order}
    used: dict[str, int] = collections.Counter()
    per_source_text: dict[str, list[str]] = collections.defaultdict(list)
    per_source_rows: dict[str, list[int]] = collections.defaultdict(list)
    total = 0

    # Least-tokens-first, NOT round-robin over conversations. One conversation per
    # source per pass looks fair and is not: a SWE trajectory is ~100k tokens and a
    # chat turn ~1k, so equal conversation counts hand the long-context sources two
    # orders of magnitude more of the budget. Measured on the first attempt, the
    # eight agent/SWE sources took ~70% of 5M while broad-domain got 0.03%.
    #
    # Repeatedly extending whichever source has the FEWEST tokens so far equalises
    # by token instead, which is the unit the Hessians actually see: short-row
    # sources contribute many conversations, long-row sources a few.
    print(f"least-tokens-first over {len(order)} sources, cap {cap:,} tokens/source ...",
          flush=True)
    while total < args.budget_tokens:
        live = [
            s for s in order
            if cursors[s] < len(by_source[s]) and used[s] < cap
        ]
        if not live:
            print("  exhausted every source before the budget was met", flush=True)
            break
        s = min(live, key=lambda k: (used[k], k))
        i = by_source[s][cursors[s]]
        cursors[s] += 1
        text = render(tok, rows[i], failures)
        if not text:
            continue
        ntok = len(tok.encode(text))
        per_source_text[s].append(text)
        per_source_rows[s].append(i)
        used[s] += ntok
        total += ntok

    if total == 0:
        raise SystemExit(
            "produced 0 calibration tokens -- every row failed to render. "
            f"failures: {dict(failures)}"
        )

    cal = out / "corpus.cal.txt"
    with open(cal, "w") as f:
        for s in order:
            for t in per_source_text.get(s, []):
                f.write(t)
                f.write("\n")
    for s, texts in per_source_text.items():
        slug = s.replace("/", "_").replace(":", "_")
        with open(out / f"corpus.cal.{slug}.txt", "w") as f:
            for t in texts:
                f.write(t + "\n")

    # Held-out eval corpora, one distribution per file. `agentic` is the slice the
    # tool-call benchmark scores, so it must never appear above.
    AGENTIC_PREFIXES = ("swe", "logminer", "dataclaw", "txt360-agent")
    slices: dict[str, list[str]] = collections.defaultdict(list)
    budgets: dict[str, int] = collections.Counter()

    # Shuffle before taking. The pack is grouped by source, so walking the test
    # indices in order starts inside one long-context source and three or four
    # 100k-token sessions exhaust the budget -- the first run produced a 440k-token
    # "distribution" made of 9 conversations. Balance by token within each slice
    # for the same reason as the calibration corpus above.
    ev_by_source: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i in test_i:
        src = rows[i].get("source") or "(none)"
        name = "agentic" if src.startswith(AGENTIC_PREFIXES) else "general"
        ev_by_source[(name, src)].append(i)
    for v in ev_by_source.values():
        rng.shuffle(v)
    ev_cursors = {k: 0 for k in ev_by_source}
    ev_used: dict[tuple[str, str], int] = collections.Counter()

    slice_budget = {"agentic": args.eval_agentic_tokens, "general": args.eval_tokens}
    for name in ("agentic", "general"):
        keys = [k for k in ev_by_source if k[0] == name]
        while budgets[name] < slice_budget[name]:
            live = [k for k in keys if ev_cursors[k] < len(ev_by_source[k])]
            if not live:
                break
            k = min(live, key=lambda kk: (ev_used[kk], kk[1]))
            i = ev_by_source[k][ev_cursors[k]]
            ev_cursors[k] += 1
            text = render(tok, rows[i], failures)
            if not text:
                continue
            ntok = len(tok.encode(text))
            slices[name].append(text)
            budgets[name] += ntok
            ev_used[k] += ntok

    for name, texts in slices.items():
        with open(out / f"corpus.eval.{name}.txt", "w") as f:
            for t in texts:
                f.write(t + "\n")

    # Records, not just rendered text: the .txt is what llama-imatrix and the PTQ
    # path consume, but publishing the selection as parquet in the same shape as
    # the pack's other tiers lets it be re-rendered against a different template
    # or tokenizer, and audited row by row.
    if args.emit_records:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rec_dir = out / "records"
        rec_dir.mkdir(exist_ok=True)
        chosen = [i for s_ in order for i in per_source_rows.get(s_, [])]
        chosen.sort()
        # Match the published tiers exactly -- list<string>, and an n_chars column
        # equal to sum(len(m) for m in messages), verified against ctx-128k-v65536.
        # large_string is unnecessary here: that exists for the full 5 GiB pack,
        # whereas these shards are ~1 MB of messages each, far under Arrow's 2 GiB
        # int32 offset cap.
        schema = pa.schema([
            ("messages", pa.list_(pa.string())),
            ("tools", pa.string()),
            ("source", pa.string()),
            ("n_chars", pa.int64()),
        ])
        per_shard = max(1, math.ceil(len(chosen) / args.shards))
        n_shards = math.ceil(len(chosen) / per_shard)
        for sh in range(n_shards):
            part = chosen[sh * per_shard : (sh + 1) * per_shard]
            tbl = pa.Table.from_pydict(
                {
                    "messages": [rows[i]["messages"] for i in part],
                    "tools": [rows[i].get("tools") for i in part],
                    "source": [rows[i].get("source") for i in part],
                    "n_chars": [
                        sum(len(m) for m in rows[i]["messages"]) for i in part
                    ],
                },
                schema=schema,
            )
            pq.write_table(
                tbl,
                rec_dir / f"train-{sh:05d}-of-{n_shards:05d}.parquet",
                compression="zstd",
            )
        (rec_dir / "tier_meta.json").write_text(json.dumps({
            "label": args.label,
            "note": "Calibration selection for GPTQ / imatrix. Drawn ONLY from the "
                    "train slice of the seeded split, so the test holdout stays "
                    "clean for evaluation. Token-balanced across sources.",
            "max_tokens": args.ctx,
            "n": len(chosen),
            "tokens": total,
            "fit_pct": 100,
            "tokenizer_note": f"{args.model} (65,536-token restricted vocabulary)",
        }, indent=2))
        print(f"records: {len(chosen):,} rows -> {n_shards} parquet shards in {rec_dir}")

    audit = {
        "pack": str(args.pack),
        "tokenizer": str(args.model),
        "rows_total": n,
        "split": {"train": len(train_i), "eval": len(eval_i), "test": len(test_i)},
        "calibration": {
            "budget_tokens": args.budget_tokens,
            "actual_tokens": total,
            "ctx": args.ctx,
            "windows_at_ctx": total // args.ctx,
            "conversations": sum(len(v) for v in per_source_text.values()),
            "max_source_share": args.max_source_share,
            "per_source_tokens": dict(sorted(used.items(), key=lambda kv: -kv[1])),
            "per_source_share": {
                s: round(v / total, 4) for s, v in sorted(used.items(), key=lambda kv: -kv[1])
            },
        },
        "eval": {
            k: {
                "tokens": budgets[k],
                "conversations": len(v),
                "budget": slice_budget.get(k),
                "sources": len({kk[1] for kk in ev_used if kk[0] == k and ev_used[kk]}),
            }
            for k, v in slices.items()
        },
        "render_failures": dict(failures),
    }
    (out / "corpora_audit.json").write_text(json.dumps(audit, indent=2))

    print(f"\ncalibration: {total:,} tokens over "
          f"{sum(len(v) for v in per_source_text.values()):,} conversations "
          f"-> {total // args.ctx} windows at ctx {args.ctx}")
    if total // args.ctx < 128:
        print(f"  WARNING: only {total // args.ctx} windows at ctx {args.ctx}. GPTQ Hessians "
              f"want >=128 samples; raise --budget-tokens or lower --ctx.")
    for name in sorted(budgets):
        print(f"eval.{name}: {budgets[name]:,} tokens, {len(slices[name]):,} conversations")
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
