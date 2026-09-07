#!/usr/bin/env python3
"""Does stage 0 fail to call tools, or just fail to do it inside the budget?

On the step-3000 run stage 0 scored 0 valid tool calls out of 36 turns while
checkpoint-3000 scored 62 of 89 -- but stage 0 also spent ~80 s/turn, which at
this card's decode rate is the full 1,536-token cap on EVERY turn, and 3 of its
4 parse failures were `unterminated string`: cut off mid-call.

Those are two different findings and the eval cannot tell them apart:

  "cannot produce a well-formed tool call"      -> a real capability gap
  "cannot finish one within 1,536 tokens"       -> a budget artifact, and the
                                                   measurement is simply wrong

This replays the same prefixes at a much larger budget and reports where the
generation actually ends. If calls appear at 4,096 that did not appear at
1,536, the cap was the finding and has to be raised before anything is quoted.

    PYTHONPATH=src python scripts/diag_toolcall_budget.py \\
        --holdout out/e4b-v65536/eval/toolcall_holdout_quick.jsonl -n 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.eval.toolcall import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT, maybe_inject_system, strip_for_api,
)

STAGE0 = "/workspace/models/gemma4-e4b-stage0-32k-v65536/final"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--holdout", type=Path, required=True)
    p.add_argument("--model", default=STAGE0)
    p.add_argument("--adapter", default=None)
    p.add_argument("-n", "--n-sessions", type=int, default=6)
    p.add_argument("--budgets", type=int, nargs="+", default=[1536, 4096])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    from quant_tuner.eval.local_gemma4 import LocalGemma4Client

    sessions = [json.loads(l) for l in a.holdout.open()][: a.n_sessions]
    prompts = []
    for s in sessions:
        msgs = maybe_inject_system(s["messages"], DEFAULT_SYSTEM_PROMPT)
        i = next((k for k, m in enumerate(msgs)
                  if m.get("role") == "assistant" and m.get("tool_calls")), None)
        if i is None:
            continue
        truth = msgs[i]["tool_calls"][0]["function"]["name"]
        prompts.append((s["session_id"], strip_for_api(msgs[:i]), s["tools"], truth))
    print(f"{len(prompts)} prefixes from {a.holdout.name}", flush=True)

    client = LocalGemma4Client(a.model, adapter=a.adapter, device=a.device,
                               max_len=65536)
    rows = []
    for budget in a.budgets:
        print(f"\n=== max_tokens = {budget} ===", flush=True)
        print(f"{'session':>14} {'out':>5} {'end':>9} {'calls':>5} {'name matches':>13} {'parse_error':>22}",
              flush=True)
        for sid, prefix, tools, truth in prompts:
            t = time.time()
            r = client.chat.completions.create(
                messages=prefix, tools=tools, max_tokens=budget, temperature=0.0)
            ch = r.choices[0]
            tcs = ch.message.tool_calls or []
            names = [tc.function.name for tc in tcs]
            rows.append({"session": sid, "budget": budget,
                         "n_out": r.usage.completion_tokens,
                         "finish": ch.finish_reason, "calls": names,
                         "truth": truth, "secs": time.time() - t})
            print(f"{sid:>14} {r.usage.completion_tokens:5d} {ch.finish_reason:>9} "
                  f"{len(tcs):5d} {str(truth in names):>13} "
                  f"{str(getattr(ch.message, 'parse_error', '') or '-'):>22}",
                  flush=True)
        got = sum(1 for x in rows if x["budget"] == budget and x["calls"])
        hit = sum(1 for x in rows if x["budget"] == budget and x["truth"] in x["calls"])
        print(f"  -> {got}/{len(prompts)} produced any call, {hit} matched the truth name",
              flush=True)

    if a.out:
        a.out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")
    print("\nREADING: if the larger budget produces calls the smaller one did not, "
          "1,536 was measuring the cap and must be raised before quoting anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
