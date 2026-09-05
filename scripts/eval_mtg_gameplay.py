#!/usr/bin/env python3
"""Score MTG gameplay decisions: exact match on the chosen move index.

Each held-out row is one decision point -- a board state plus a numbered list
of legal moves -- answered with `choose_move(move_index, rationale)`. The index
is the whole ground truth, so this is exact match, and it does NOT go through
`quant_tuner.eval.toolcall`: that harness's `param_acc` averages over every
argument, and `rationale` is free text that will never match, so a perfect
player and a random one would score within noise of each other there.

Three things this reports that a bare accuracy number would hide:

* `chance` -- the mean of 1/n_legal_moves over the scored rows. Decision points
  differ wildly in how many options they offer (a turn-1 land drop offers two
  or three, a developed board offers twenty), so raw accuracy is not comparable
  between models unless they answered the same rows, and is not interpretable
  at all without this baseline.
* `well_formed` / `in_range` -- a model can fail by not emitting a parseable
  `choose_move` at all, or by naming an index outside the list it was shown.
  Those are different failures from choosing badly, and the system prompt
  explicitly forbids the second ("Never pick an index outside the list").
* accuracy bucketed by game turn -- the board grows monotonically through a
  game, so late turns carry longer prompts AND more legal moves. This is the
  cheapest read available on whether the model keeps hold of a goal as the
  context grows, and it is paired: every model answers the same rows.

    PYTHONPATH=src python scripts/eval_mtg_gameplay.py \
        --holdout /workspace/mtg-gameplay-heldout.jsonl \
        --adapters .../checkpoint-3000 --out out/mtg_gameplay.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BASE = "/workspace/models/gemma4-e4b-qat-v65536-text"
STAGE0 = "/workspace/models/gemma4-e4b-stage0-32k-v65536/final"

# "Turn 5, main step of your turn." and the legal-move list, which is
# ZERO-indexed with a colon ("0: Play Ghost Quarter") and sits at the very end
# of the user turn behind a fixed marker. Anchor on the marker: the board state
# above it also contains lines that begin with digits.
RE_TURN = re.compile(r"\bTurn\s+(\d+)", re.I)
RE_OPT = re.compile(r"^(\d+):\s+\S", re.M)
MARKER = "choose_move` with the index"


def _j(x):
    return json.loads(x) if isinstance(x, str) else x


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        msgs = [_j(m) for m in r["messages"]]
        a = next((m for m in msgs if m.get("role") == "assistant"
                  and m.get("tool_calls")), None)
        if a is None:
            continue
        args = a["tool_calls"][0]["function"]["arguments"]
        args = _j(args) if isinstance(args, str) else args
        truth = args.get("move_index")
        if not isinstance(truth, int):
            continue
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "") or ""
        tail = user.rsplit(MARKER, 1)[-1] if MARKER in user else user
        opts = [int(m.group(1)) for m in RE_OPT.finditer(tail)]
        turn = int(RE_TURN.search(user).group(1)) if RE_TURN.search(user) else 0
        rows.append({
            "row": r.get("_row"),
            "prefix": [m for m in msgs if m.get("role") in ("system", "user")],
            "tools": _j(r["tools"]),
            "truth": truth,
            "n_opts": max(opts) - min(opts) + 1 if opts else 0,
            "lo": min(opts) if opts else 0,
            "hi": max(opts) if opts else 0,
            "turn": turn,
        })
    return rows


def score(client, rows, max_tokens, temperature, label, batch_size) -> dict:
    """Batched scoring, longest prompts first.

    A batch runs until its slowest member stops, so mixing a turn-1 board with
    a turn-15 one pays the long prompt's price for both. Sorting by length
    before batching keeps each batch roughly uniform; the results are unsorted
    back to the holdout's order so per-row records stay paired across models.
    """
    t0 = time.time()
    order = sorted(range(len(rows)),
                   key=lambda i: -len(json.dumps(rows[i]["prefix"])))
    reqs = [{"messages": rows[i]["prefix"], "tools": rows[i]["tools"]}
            for i in order]
    parsed = client.generate_batch(reqs, temperature=temperature,
                                   max_tokens=max_tokens, batch_size=batch_size)
    by_row = dict(zip(order, parsed))

    per = []
    for i, r in enumerate(rows):
        pr = by_row[i]
        pick, well = None, False
        for tc in pr["tool_calls"]:
            if tc["function"]["name"] != "choose_move":
                continue
            try:
                v = json.loads(tc["function"]["arguments"]).get("move_index")
            except json.JSONDecodeError:
                v = None
            if isinstance(v, int):
                pick, well = v, True
                break
        per.append({
            "row": r["row"], "turn": r["turn"], "n_opts": r["n_opts"],
            "truth": r["truth"], "pick": pick, "well_formed": well,
            "truncated": bool(pr.get("truncated_thought")
                              or pr.get("_n_out", 0) >= max_tokens),
            "in_range": bool(well and r["lo"] <= pick <= r["hi"]),
            "correct": bool(well and pick == r["truth"]),
        })
    dt = time.time() - t0
    print(f"  [{label}] {len(rows)} rows in {dt/60:.1f} min "
          f"({dt/len(rows):.1f}s/row)", flush=True)
    return {"label": label, "per_row": per, "secs": dt}


def summarise(run, rows) -> dict:
    per = run["per_row"]
    n = len(per)
    chance = sum(1.0 / r["n_opts"] for r in rows if r["n_opts"]) / max(
        1, sum(1 for r in rows if r["n_opts"]))
    buckets: dict[str, list] = {}
    for p in per:
        b = ("t1-4" if p["turn"] <= 4 else "t5-9" if p["turn"] <= 9
             else "t10-14" if p["turn"] <= 14 else "t15+")
        buckets.setdefault(b, []).append(p)
    return {
        "n": n,
        "acc": sum(p["correct"] for p in per) / n,
        "chance": chance,
        "well_formed": sum(p["well_formed"] for p in per) / n,
        "truncated": sum(p.get("truncated", False) for p in per) / n,
        "in_range": sum(p["in_range"] for p in per) / n,
        "by_turn": {k: {"n": len(v),
                        "acc": sum(p["correct"] for p in v) / len(v)}
                    for k, v in sorted(buckets.items())},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--holdout", type=Path,
                   default=Path("/workspace/mtg-gameplay-heldout.jsonl"))
    p.add_argument("--base", default=STAGE0,
                   help="Frozen base the adapters sit on (default: stage 0 final)")
    p.add_argument("--pruned-base", default=BASE)
    p.add_argument("--include-pruned-base", action="store_true")
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--out", type=Path, required=True)
    # Ground-truth reasoning + call on this slice is a median of 365 tokens and
    # a p90 of 1,272 (est. at 3.2 chars/token). At 256 the eval would mostly be
    # measuring where the budget ran out; `truncated` in the summary is there to
    # catch it if this is still too low.
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-len", type=int, default=32768)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Prompts per generate() call. Decode is bandwidth-bound, "
                        "so this is close to a free speedup until VRAM binds.")
    a = p.parse_args()

    from quant_tuner.eval.local_gemma4 import LocalGemma4Client

    rows = load(a.holdout)
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} decision points; median options "
          f"{sorted(r['n_opts'] for r in rows)[len(rows)//2]}", flush=True)

    arms = []
    if a.include_pruned_base:
        arms.append(("pruned base", a.pruned_base, None))
    arms.append(("stage 0 final", a.base, None))
    for ad in a.adapters:
        arms.append((Path(ad.rstrip("/")).name, a.base, ad))

    out = {"holdout": str(a.holdout), "n": len(rows), "runs": []}
    for label, base, adapter in arms:
        client = LocalGemma4Client(base, adapter=adapter, device=a.device,
                                   max_len=a.max_len)
        run = score(client, rows, a.max_tokens, a.temperature, label,
                    a.batch_size)
        run["summary"] = summarise(run, rows)
        out["runs"].append(run)
        s = run["summary"]
        print(f"\n{label:>22}  acc {s['acc']:.3f} (chance {s['chance']:.3f})  "
              f"well_formed {s['well_formed']:.3f}  in_range {s['in_range']:.3f}",
              flush=True)
        del client
        import gc, torch
        gc.collect()
        if a.device != "cpu":
            torch.cuda.empty_cache()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")

    if len(out["runs"]) > 1:
        ref = out["runs"][0]
        print(f"\n{'model':>22} {'acc':>7} {'vs ref':>8} {'flips +/-':>12}")
        for run in out["runs"]:
            s = run["summary"]
            up = sum(1 for x, y in zip(run["per_row"], ref["per_row"])
                     if x["correct"] and not y["correct"])
            dn = sum(1 for x, y in zip(run["per_row"], ref["per_row"])
                     if not x["correct"] and y["correct"])
            d = s["acc"] - ref["summary"]["acc"]
            print(f"{run['label']:>22} {s['acc']:7.3f} {d:+8.3f} {f'+{up}/-{dn}':>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
