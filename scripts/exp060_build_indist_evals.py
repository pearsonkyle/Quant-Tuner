"""Build the in-distribution eval corpora for exp-060 from held-out data.

The handoff declared the ``tools`` and ``agentic`` eval columns unavailable, because the
on-disk agent logs are local-only. They are recoverable after all: ``sft.jsonl.gz`` carries
a ``holdout`` split (disjoint from the ``train`` split we calibrate on) with 23 CLI-log +
54 agent-log sessions and 8 SWE trajectories.

Corpora written (each is a SEPARATE distribution and needs its OWN ``kld.build_baseline``
— never concatenate them):

* ``corpus.eval.tools.txt``   — sft holdout, ``logs`` + ``logs-agents``. In-distribution
  tool-calling fidelity, genuinely disjoint from calibration.
* ``corpus.eval.agentic.txt`` — sft holdout, ``swe-trajectories``.
* ``corpus.eval.broad.txt``   — sft holdout, ``broad-instruct``.
* ``corpus.eval.redteam.txt`` — sft holdout, ``redteam-refusals`` (refusal behavior is what
  low-bit quantization erodes first).
* ``corpus.eval.cal8k.txt``   — a deterministic slice of the TRANSFERRED
  ``corpus.cal.jsonl.gz``. **This one is a fit measure, not a holdout**: it is the published
  build's *calibration* corpus, drawn from the logs train slice, and our sft calibration
  also uses ``split == "train"``, so the two share source sessions. Report it as
  in-distribution fidelity against the published calibration distribution — never as
  generalization.

Eval ctx is 8192, NOT the 32768 the calibration corpus is packed for. Those are different
concerns: 32768 is a *packing* parameter chosen so a whole agentic trajectory fits one
calibration context, while eval ctx only sets PPL/KLD chunking. 8192 gives ~4x more chunks
per corpus (materially more stable statistics on a 150k-token corpus) and keeps the headline
numbers on the same footing as the published release.

    PYTHONPATH=src .venv/bin/python scripts/exp060_build_indist_evals.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data import ingest, reasoning, split
from quant_tuner.data.universal import clip_tool_messages

SEED = 42
EVAL_CTX = 8192
EVAL_CTX_HEADROOM = 692
SYSTEM_PROSE_BUDGET = 256
MAX_TOOL_OUTPUT_TOKENS = 512
REASONING_POLICY = "auto"

# stratum -> (sft sources, target tokens)
SPECS = {
    "tools":   (("logs", "logs-agents"), 150_000),
    "agentic": (("swe-trajectories",), 100_000),
    "broad":   (("broad-instruct",), 100_000),
    "redteam": (("redteam-refusals",), 30_000),
}
CAL8K_TARGET_TOKENS = 150_000


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sft", type=Path, default=Path("/workspace/sft.jsonl.gz"))
    p.add_argument("--cal-jsonl", type=Path,
                   default=Path("/workspace/corpus.cal.jsonl.gz"))
    p.add_argument("--run", default="exp-060-32k")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    a = p.parse_args()

    cap = max(512, EVAL_CTX - EVAL_CTX_HEADROOM)
    out = REPO / "out" / a.run / "corpora"
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)

    rows = [json.loads(ln) for ln in gzip.open(a.sft, "rt")]
    holdout = [r for r in rows if r.get("split") == "holdout"]
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for r in holdout:
        by_source[r.get("source", "")].append(r)

    pack_kwargs = dict(
        per_session_cap=cap, seed=SEED,
        system_prose_budget=SYSTEM_PROSE_BUDGET, full_prose_quota=1,
        max_windows_per_session=8, tool_schema_quota=1, user_anchor=True,
    )

    audit: dict = {"eval_ctx": EVAL_CTX, "window_cap": cap, "seed": SEED, "corpora": {}}
    # llama-perplexity refuses a file shorter than TWO contexts ("you need at least
    # 2*n_ctx tokens"). It exits 0 having written only a 12-byte header, so the pipeline's
    # existence-based step() records a bogus baseline and the failure only surfaces later,
    # at bench time, as a dead KLD column. Catch it here instead.
    min_tokens = 2 * EVAL_CTX
    undersized: list[str] = []

    for name, (sources, target) in SPECS.items():
        sessions: list[dict] = []
        for s in sources:
            sessions.extend(by_source.get(s, []))
        if not sessions:
            print(f"  !! {name}: no holdout sessions for {sources} — skipping")
            continue
        prepped = []
        for s in sessions:
            s = dict(s)
            msgs = ingest.normalize_messages(s.get("messages") or [])
            msgs = reasoning.apply_policy(msgs, REASONING_POLICY)
            msgs, _n = clip_tool_messages(msgs, tok, MAX_TOOL_OUTPUT_TOKENS)
            s["messages"] = msgs
            prepped.append(s)
        chunks, _kept, total, pack = split.stratified_pack(
            prepped, tok, target_tokens=target, **pack_kwargs)
        path = out / f"corpus.eval.{name}.txt"
        split.write_corpus(chunks, path)
        audit["corpora"][name] = {
            "path": str(path), "source_slice": f"sft holdout: {', '.join(sources)}",
            "sessions": len(prepped), "windows": len(chunks), "tokens": total,
            "target_tokens": target, "pack_audit": pack,
            "caveat": "llama-perplexity has no --parse-special: chat markers tokenize as "
                      "plain BPE. Quant-vs-quant only, not absolute PPL.",
        }
        if total < min_tokens:
            undersized.append(f"{name} ({total:,} < {min_tokens:,})")
        print(f"  {name:10s} {len(chunks):4d} windows / {total:,} tokens "
              f"({len(prepped)} sessions)"
              + ("   !! UNDERSIZED for eval_ctx" if total < min_tokens else ""))

    # --- cal8k: a slice of the TRANSFERRED calibration corpus -------------------------
    if a.cal_jsonl.exists():
        recs = [json.loads(ln) for ln in gzip.open(a.cal_jsonl, "rt")]
        recs.sort(key=lambda r: r["i"])
        # Deterministic even stride across the whole file, so every source is represented
        # rather than whichever ones happen to sit at the head.
        picked: list[str] = []
        total = 0
        stride = max(1, len(recs) // 60)
        for r in recs[::stride]:
            if total >= CAL8K_TARGET_TOKENS:
                break
            picked.append(r["text"])
            total += len(tok(r["text"], add_special_tokens=False)["input_ids"])
        path = out / "corpus.eval.cal8k.txt"
        split.write_corpus(picked, path)
        audit["corpora"]["cal8k"] = {
            "path": str(path), "source": str(a.cal_jsonl),
            "windows": len(picked), "tokens": total, "stride": stride,
            "MEASURES": "FIT, NOT GENERALIZATION — this is the published build's "
                        "calibration corpus (logs train slice); our sft calibration uses "
                        "the same train split, so the two share source sessions.",
        }
        print(f"  {'cal8k':10s} {len(picked):4d} windows / {total:,} tokens "
              f"(fit measure, overlaps calibration)")

    audit["min_tokens_for_eval_ctx"] = min_tokens
    audit["undersized"] = undersized
    (out / "indist_eval_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"\nwrote {out / 'indist_eval_audit.json'}")
    if undersized:
        print(f"\n!! UNDERSIZED for eval_ctx={EVAL_CTX}: {', '.join(undersized)}")
        print("   llama-perplexity needs >= 2*n_ctx tokens; it will exit 0 after writing a")
        print("   12-byte header, and the pipeline's existence-based step() will treat that")
        print("   empty file as a valid baseline. Either drop these evals from --evals or")
        print("   bench them separately at a smaller --eval-ctx with their own baselines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
