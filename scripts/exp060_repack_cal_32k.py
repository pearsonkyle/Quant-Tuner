"""Repack the exp-060 calibration corpus at ctx=32768 from ``sft.jsonl.gz``.

Why this exists
---------------
The corpus that was transferred prebuilt (``corpus.cal.jsonl.gz``) is PACKED for ctx=8192:
``window_cap = 8192 - 692 = 7500``, and measured on the file the longest window is 11,390
tokens with 29% of windows sitting at the cap. Reading that file at ``-c 32768`` does not
calibrate on 32K trajectories — it glues ~4 unrelated windows into each context. The
trajectories were already cut when they were packed, and no ctx setting recovers context
that is not in the file.

``sft.jsonl.gz`` is the other half of what ``data.universal`` writes: the FULL, unpacked
conversations (``messages`` + real ``tools``), with a ``split`` field matching the
calibration corpus. That IS repackable at any window size, so this script rebuilds the
calibration corpus at 32768 using the repo's own packer rather than a reimplementation.

What this reproduces, and what it cannot
----------------------------------------
Reproduced from sft (all use ``split == "train"``, keeping the eval holdouts held out):

* ``logs``            — sft sources ``logs`` + ``logs-agents``
* ``reasoning``       — ``universal.reasoning_windows`` over those same log sessions, so
  reasoning still lands in the one position a chat template preserves (final assistant turn)
* ``swe-trajectories``
* ``broad``           — sft source ``broad-instruct``
* ``redteam-refusals``
* ``wiki``            — optional, ``--wiki``; fetched separately (not in sft)

Deviations from the documented 8192 build, which are real and must be reported with any
number produced from this corpus:

1. **System prompts are scrubbed.** sft rows carry ``system_scrub``; the calibration corpus
   is deliberately NOT scrubbed (the packer's ``system_prose_budget`` stub is the intended
   mechanism). The packer's stub still applies on top.
2. ``broad-instruct`` is the sft form of the broad supplement, not byte-identical to the
   ``broad-supplement`` slice of the documented corpus.
3. Numbers from this corpus are **not comparable** to the published 8192 ladder, in either
   direction — different packing AND different ctx.

    PYTHONPATH=src .venv/bin/python scripts/exp060_repack_cal_32k.py --ctx 32768
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data import ingest, reasoning, split, universal
from quant_tuner.data.universal import clip_tool_messages, reasoning_windows

# sft `source` -> the corpus stratum it feeds (mirrors data/universal.py's part names).
SOURCE_MAP = {
    "logs": "logs",
    "logs-agents": "logs",
    "swe-trajectories": "swe-trajectories",
    "broad-instruct": "broad-supplement",
    "redteam-refusals": "redteam-refusals",
}

# UniversalConfig defaults — kept in sync deliberately; see data/universal.py.
SEED = 42
CTX_HEADROOM = 692
SYSTEM_PROSE_BUDGET = 256
FULL_PROSE_QUOTA = 1
MAX_WINDOWS_PER_SESSION = 8
MAX_WINDOWS_PER_SWE_SESSION = 32
MAX_REASONING_WINDOWS_PER_SESSION = 4
TOOL_SCHEMA_QUOTA = 1
MAX_TOOL_OUTPUT_TOKENS = 512
REASONING_POLICY = "auto"

# Token budgets per stratum (UniversalConfig; None = uncapped).
BUDGETS = {
    "logs": 2_000_000,
    "reasoning": 1_000_000,
    "swe-trajectories": 1_000_000,
    "broad-supplement": None,
    "redteam-refusals": None,
    "wiki": None,
}
UNCAPPED = 1 << 40


def resolve_budgets(
    budget_specs: list[str], drop_specs: list[str],
) -> tuple[dict[str, int | None], set[str]]:
    """Apply --budget/--drop overrides to the module defaults.

    Split out of ``main`` so the "no flags => the pinned exp-060 corpus" property
    is testable. That property is what lets this one script build both the pinned
    corpus and a re-cut one; if a default drifts, every number attributed to the
    original corpus silently stops being reproducible.

    Raises ``ValueError`` on a malformed spec or an unknown stratum — an unknown
    name must not be a silent no-op, or a typo'd ``--drop redteam_refusals``
    would leave the stratum in and quietly contradict the audit.
    """
    budgets: dict[str, int | None] = dict(BUDGETS)
    for spec in budget_specs:
        name, sep, val = spec.partition("=")
        if not sep:
            raise ValueError(f"--budget expects NAME=TOKENS, got {spec!r}")
        if name not in budgets:
            raise ValueError(f"unknown stratum {name!r}; known: {sorted(budgets)}")
        budgets[name] = None if val.lower() == "none" else int(val)
    dropped = set(drop_specs)
    if unknown := dropped - set(budgets):
        raise ValueError(
            f"--drop of unknown stratum(s) {sorted(unknown)}; known: {sorted(budgets)}")
    return budgets, dropped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sft", type=Path, default=Path("/workspace/sft.jsonl.gz"))
    p.add_argument("--run", default="exp-060-32k")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--split", default="train")
    p.add_argument("--wiki", type=Path, default=None,
                   help="raw wiki text file (wiki.test.raw); omitted if not given")
    p.add_argument("--max-reasoning-windows", type=int,
                   default=MAX_REASONING_WINDOWS_PER_SESSION,
                   help="windows cut per session that END on a reasoning turn. Measured on "
                        "this sft train split: 4 -> 467 non-empty <think> blocks (budget "
                        "underspent at 638k of 1M tokens); 16 -> 1,330 and the budget fills; "
                        "past 16 a few sessions absorb the whole budget and diversity drops.")
    p.add_argument("--chat-template", type=Path, default=None,
                   help="render the corpus with this .jinja instead of the tokenizer's own. "
                        "The corpus IS chat-templated text, so whatever renders it is what "
                        "the quantizer sees — pin it here rather than trusting whichever "
                        "template the HF snapshot happens to carry.")
    p.add_argument("--drop", action="append", default=[], metavar="STRATUM",
                   help="omit a stratum entirely (repeatable). Accepts any conversation "
                        "stratum plus `reasoning`.")
    p.add_argument("--budget", action="append", default=[], metavar="NAME=TOKENS",
                   help="override a stratum's token budget (repeatable); `none` = uncapped.")
    a = p.parse_args()

    try:
        budgets, dropped = resolve_budgets(a.budget, a.drop)
    except ValueError as e:
        p.error(str(e))

    window_cap = max(512, a.ctx - CTX_HEADROOM)
    out = REPO / "out" / a.run / "corpora"
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.chat_template:
        # Assigning the attribute is what apply_chat_template reads; there is no
        # need to touch tokenizer_config.json (whose `chat_template` key would
        # otherwise win — see vocab.py's converter precedence).
        tok.chat_template = a.chat_template.read_text()
        print(f"chat template: {a.chat_template} ({len(tok.chat_template):,} chars)")

    opener = gzip.open if a.sft.suffix == ".gz" else open
    rows = [json.loads(ln) for ln in opener(a.sft, "rt")]
    rows = [r for r in rows if r.get("split") == a.split]
    by_stratum: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        stratum = SOURCE_MAP.get(r.get("source", ""))
        if stratum is None:
            continue
        # stratified_pack strata on `source`; keep the finer sft label so the packer's
        # round-robin still spreads the budget across cli-vs-agent logs.
        by_stratum[stratum].append(r)
    print(f"sft rows in split={a.split}: {len(rows)}")
    for k, v in sorted(by_stratum.items()):
        print(f"  {k:20s} {len(v):5d} sessions")

    def prepare(sessions: list[dict]) -> list[dict]:
        """Exactly universal.build's prepare(): normalize, reasoning policy, clip tools."""
        out_s = []
        for s in sessions:
            s = dict(s)
            msgs = ingest.normalize_messages(s.get("messages") or [])
            msgs = reasoning.apply_policy(msgs, REASONING_POLICY)
            msgs, _n = clip_tool_messages(msgs, tok, MAX_TOOL_OUTPUT_TOKENS)
            s["messages"] = msgs
            out_s.append(s)
        return out_s

    pack_kwargs = dict(
        per_session_cap=window_cap,
        seed=SEED,
        system_prose_budget=SYSTEM_PROSE_BUDGET,
        full_prose_quota=FULL_PROSE_QUOTA,
        max_windows_per_session=MAX_WINDOWS_PER_SESSION,
        tool_schema_quota=TOOL_SCHEMA_QUOTA,
        user_anchor=True,   # Qwen's strict template refuses a window with no user turn
    )

    parts: list[tuple[str, list[str], int, dict]] = []
    audit_sources: dict = {}

    def budget(name: str) -> int:
        b = budgets.get(name)
        return UNCAPPED if b is None else b

    # --- conversation strata ----------------------------------------------------------
    log_sessions: list[dict] = []
    for stratum in ("logs", "swe-trajectories", "broad-supplement", "redteam-refusals"):
        sessions = by_stratum.get(stratum) or []
        if not sessions or stratum in dropped:
            if stratum in dropped:
                print(f"  dropped {stratum:20s} ({len(sessions)} sessions) — --drop")
            continue
        sessions = prepare(sessions)
        if stratum == "logs":
            log_sessions = sessions
        kw = dict(pack_kwargs)
        if stratum == "swe-trajectories":
            kw["max_windows_per_session"] = MAX_WINDOWS_PER_SWE_SESSION
        chunks, _kept, total, pack = split.stratified_pack(
            sessions, tok, target_tokens=budget(stratum), **kw)
        split.write_corpus(chunks, out / f"corpus.cal.{stratum}.txt")
        parts.append((stratum, chunks, total, pack))
        audit_sources[stratum] = {
            "sessions": len(sessions), "windows": len(chunks), "tokens": total,
            "target_tokens": budgets.get(stratum), "pack_audit": pack,
        }
        print(f"  packed {stratum:20s} {len(chunks):5d} windows / {total:,} tokens")

    # --- reasoning-terminal windows over the SAME log sessions ------------------------
    # Without this the corpus carries essentially no reasoning: chat templates keep
    # reasoning only on a render's FINAL assistant turn and scrub it from history.
    if log_sessions and "reasoning" not in dropped:
        rchunks: list[str] = []
        rtotal = 0
        contributing = 0
        candidates = list(log_sessions)
        random.Random(SEED).shuffle(candidates)   # else the first few sessions absorb it all
        for s in candidates:
            if rtotal >= budget("reasoning"):
                break
            wins = reasoning_windows(s, tok, window_cap,
                                     max_windows=a.max_reasoning_windows)
            if wins:
                contributing += 1
            for text, n in wins:
                if rtotal >= budget("reasoning"):
                    break
                rchunks.append(text)
                rtotal += n
        if rchunks:
            split.write_corpus(rchunks, out / "corpus.cal.reasoning.txt")
            parts.append(("reasoning", rchunks, rtotal, {}))
            audit_sources["reasoning"] = {
                "windows": len(rchunks), "tokens": rtotal,
                "sessions_contributing": contributing,
                "max_windows_per_session": a.max_reasoning_windows,
                "target_tokens": budgets["reasoning"],
                "note": "overlaps the `logs` windows by design — re-renders the same "
                        "conversations so a reasoning turn lands last",
            }
            print(f"  packed {'reasoning':20s} {len(rchunks):5d} windows / {rtotal:,} tokens")

    # --- wiki (optional; not present in sft) ------------------------------------------
    if a.wiki and a.wiki.exists():
        # split.chunk_text, NOT pack_raw_samples: wiki must enter as many window-sized
        # chunks so interleave_many spreads it through the file. As one monolith a
        # 250k-token wiki head eats a token-budgeted calibrator's entire budget.
        all_chunks = split.chunk_text(a.wiki.read_text(), approx_chars=window_cap * 4)
        wchunks, wtotal = [], 0
        wiki_budget = BUDGETS.get("wiki")
        for c in all_chunks:
            if wiki_budget is not None and wtotal >= wiki_budget:
                break
            wchunks.append(c)
            wtotal += len(tok(c, add_special_tokens=False)["input_ids"])
        if wchunks:
            split.write_corpus(wchunks, out / "corpus.cal.wiki.txt")
            parts.append(("wiki", wchunks, wtotal, {}))
            audit_sources["wiki"] = {"windows": len(wchunks), "tokens": wtotal,
                                     "chunks_available": len(all_chunks),
                                     "path": str(a.wiki)}
            print(f"  packed {'wiki':20s} {len(wchunks):5d} windows / {wtotal:,} tokens")
    elif a.wiki:
        print(f"  !! wiki file {a.wiki} not found — skipping wiki stratum")

    # --- interleave so a token-budgeted sampler sees every source ---------------------
    ordered = split.interleave_many([c for _n, c, _t, _p in parts])
    cal = out / "corpus.cal.txt"
    split.write_corpus(ordered, cal)

    # Per-window sidecar, same shape as the transferred corpus.cal.jsonl.gz: one record
    # per window with its source label. This is what you query when a number looks wrong
    # and you need to know which source a span came from — and it makes the corpus
    # portable to another box without re-running the pack.
    source_of: dict[int, str] = {}
    for name, chunks, _t, _p in parts:
        for c in chunks:
            source_of[id(c)] = name
    win_tokens: dict[str, list[int]] = collections.defaultdict(list)
    sidecar = out / "corpus.cal.jsonl.gz"
    with gzip.open(sidecar, "wt") as fh:
        for i, chunk in enumerate(ordered):
            src = source_of.get(id(chunk), "unknown")
            n = len(tok(chunk, add_special_tokens=False)["input_ids"])
            win_tokens[src].append(n)
            fh.write(json.dumps({"i": i, "source": src, "n_chars": len(chunk),
                                 "n_tokens": n, "text": chunk}) + "\n")

    total_tokens = sum(t for _n, _c, t, _p in parts)
    audit = {
        "builder": "exp060_repack_cal_32k.py (repacked from sft.jsonl.gz — the prebuilt "
                   "corpus.cal.jsonl.gz is packed for ctx=8192 and cannot be re-cut)",
        "seed": SEED,
        "ctx": a.ctx,
        "window_cap": window_cap,
        "ctx_note": f"every calibrator must read this corpus at ctx={a.ctx}; windows are "
                    f"packed to fill one context. NOT comparable to the 8192 build.",
        "source_sft": str(a.sft),
        "split": a.split,
        # The corpus is chat-templated text: whatever rendered it IS what the
        # quantizer sees, so record it next to the token counts rather than
        # leaving it implicit in whichever snapshot the tokenizer came from.
        "chat_template": str(a.chat_template) if a.chat_template else "tokenizer default",
        "budgets": budgets,
        "dropped_strata": sorted(dropped),
        "deviations_from_documented_corpus": [
            "system prompts are scrubbed (sft system_scrub); the documented calibration "
            "corpus is deliberately unscrubbed",
            "broad stratum is sft `broad-instruct`, not the `broad-supplement` slice",
            "wiki included only if --wiki was passed",
        ],
        "calibration": {
            "path": str(cal),
            "bytes": cal.stat().st_size,
            "windows": len(ordered),
            "total_tokens": total_tokens,
            "per_source": audit_sources,
            "token_share": {n: round(t / total_tokens, 4)
                            for n, _c, t, _p in sorted(parts, key=lambda x: -x[2])},
            "window_tokens": {
                src: {
                    "n": len(v), "max": max(v), "median": sorted(v)[len(v) // 2],
                    "p90": sorted(v)[int(0.9 * (len(v) - 1))],
                    "over_7500": sum(1 for n in v if n > 7500),
                    "over_16384": sum(1 for n in v if n > 16384),
                }
                for src, v in sorted(win_tokens.items()) if v
            },
            "sidecar": str(sidecar),
        },
    }

    # Measured on the bytes as written — newline="" so the \r in agent tool output is not
    # silently rewritten (universal.scan_special_tokens guards the same way).
    with cal.open(newline="") as f:
        text = f.read()
    from quant_tuner.data.template_check import (
        KNOWN_TOOL_CALL_MARKERS,
        KNOWN_TOOL_RESPONSE_MARKERS,
    )
    calls = universal.tool_call_marker_counts(text, KNOWN_TOOL_CALL_MARKERS)
    resps = universal.tool_call_marker_counts(text, KNOWN_TOOL_RESPONSE_MARKERS)
    audit["calibration"]["tool_calls"] = {
        "tool_call_markers": calls,
        "tool_call_marker_total": sum(calls.values()),
        "tool_response_markers": resps,
        "tool_response_marker_total": sum(resps.values()),
    }
    (out / "corpora_audit.json").write_text(json.dumps(audit, indent=2))

    c = audit["calibration"]
    print(f"\nwrote {cal}")
    print(f"  windows      : {c['windows']:,}")
    print(f"  bytes        : {c['bytes']:,}")
    print(f"  total_tokens : {c['total_tokens']:,}")
    print(f"  token_share  : {c['token_share']}")
    print(f"  tool calls   : {c['tool_calls']['tool_call_marker_total']:,}")
    print(f"wrote {out / 'corpora_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
