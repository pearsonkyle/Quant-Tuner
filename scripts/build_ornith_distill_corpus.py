"""iter-5 QAT corpus: distill VERIFIED SWE-rebench solutions from a strong solver.

The iter-2/3/4 corpus was scraped Claude-Code / Gemini-CLI agent logs — imitation
data, not outcome data. iter-4 proved the cost: driving the masked loss lower on that
corpus made the agent tidier but *lowered* patch rate. The lever the audit and the
iter-4 verdict both point at is BETTER DATA — verified successful trajectories.

This builds exactly that. It takes the agentic SWE-rebench trajectories a strong model
(Ornith-1.0-9B at >=Q5_K_M: 100% patch, 60% resolved on the 10-instance holdout) already
produced, keeps ONLY the instances that actually RESOLVED (passed the hidden
FAIL_TO_PASS / PASS_TO_PASS tests — a patch that doesn't pass is the same mimicry trap),
and re-renders each trajectory through *Bonsai's* chat template + tokenizer with the
assistant/tool spans masked. The student never sees Ornith's tokens — it learns to
imitate the SHAPE of a winning agentic trajectory in its own vocabulary.

Why this can move capability where the log corpus couldn't:
  * Outcome-quality: every trajectory ends in a graded PASS, so the tokens trained are
    from a solution that works, not a plausible-looking diff that doesn't.
  * Patch-writing supervision: the exact thing the patch-rate metric was starving for
    (the audit's data-scaling recommendation iii).
  * Disjoint source: nebius/SWE-rebench issues never touch logtrain/wiki, so this is
    genuine generalization, not fit — and no eval-corpus contamination.

Note on distillation type: this is DATA/trajectory distillation, not logit-KD. Ornith
(Qwen3.5-VL, vocab 248,320) and Bonsai (Qwen3, vocab 151,669) have different tokenizers,
so per-token logit-KD is impossible across them. Logit-KD is a separate, composable lever
from the *same-vocab* dense parent (Qwen3-8B fp16) via the trainer's --kd-teacher.

The trajectories are the OpenAI Agents SDK item list saved by
``eval/agents/openai_agents.py`` (``<ws>/trajectories/<model>/<id>.traj.json``):
  USER(problem)  ->  [ASST_MSG(reasoning)?  CALL(bash)  ]*  OUTPUT  ...
We collapse an optional reasoning ``message`` plus the following ``function_call`` into a
single assistant turn (content + tool_calls), and each ``function_call_output`` into a
``role=tool`` message — the standard Qwen tool-calling shape. Masking, packing, the
stop-token label fix, the density audit and the fingerprint are all reused verbatim from
``build_qat_masked_corpus`` so this corpus is bit-for-bit trainer-compatible.

    PYTHONPATH=src .venv/bin/python scripts/build_ornith_distill_corpus.py \
        --traj-dir out/swe-rebench/ornith-q5km-swe/trajectories/Ornith-1.0-9B-Q5_K_M \
        --results out/swe-rebench/ornith-q5km-swe/results.csv \
        --max-tool-tokens 1024 --out out/exp-058/distill_corpus_4096.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from build_qat_masked_corpus import (  # noqa: E402  (sibling script, same dir on sys.path)
    CHAT_TEMPLATE,
    MODEL,
    corpus_fingerprint,
    masked_ids_for_session,
    pack,
    truncate_tool_messages,
    IM_END,
)

# The single bash tool the openai-agents backend registers (eval/agents/openai_agents.py):
# @function_tool def bash(command: str) -> str. The SDK derives this JSON schema from the
# signature+docstring; we render the SAME schema into Bonsai's # Tools block so the
# student trains schema-conditioned on the tool it will actually be given at eval.
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the repository checkout at /testbed "
                       "and return its combined output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

# The backend's system prompt (kept in sync with eval/agents/openai_agents._SYSTEM_PROMPT).
# It is context (masked out), but using the real one keeps the training distribution
# faithful to how Bonsai is driven at eval time.
SYSTEM_PROMPT = (
    "You are an autonomous software engineer fixing a real bug in a Python "
    "repository already checked out at /testbed (the current working directory).\n\n"
    "You have one tool: bash(command). Use it to explore the code, reproduce the "
    "issue, edit files, and run the project's tests. Make all edits directly on the "
    "files under /testbed with shell commands. Work in small steps and inspect "
    "command output before continuing.\n\n"
    "Your job is to FIX THE BUG IN THE LIBRARY/SOURCE CODE. Find the function or "
    "class responsible and change its implementation. First reproduce the failure, "
    "then edit the source, then re-run to confirm your source change makes the "
    "failing behavior pass. When the fix is complete and verified, stop and give a "
    "short summary of what you changed."
)


def _text_of(item: dict) -> str:
    """Assistant 'message' content is a list of {text, ...} blocks (or a str)."""
    c = item.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""


def trajectory_to_messages(items: list[dict]) -> list[dict]:
    """OpenAI-Agents item list -> Qwen chat messages.

    Collapses an optional reasoning ``message`` + the following ``function_call``(s)
    into one assistant turn; each ``function_call_output`` becomes a ``role=tool``
    message. The leading ``role=user`` item is the problem statement.
    """
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    pending_text = ""            # buffered assistant reasoning awaiting its call
    pending_calls: list[dict] = []

    def flush_assistant():
        nonlocal pending_text, pending_calls
        if pending_text.strip() or pending_calls:
            m: dict = {"role": "assistant", "content": pending_text}
            if pending_calls:
                m["tool_calls"] = pending_calls
            msgs.append(m)
        pending_text, pending_calls = "", []

    for it in items:
        typ = it.get("type")
        if it.get("role") == "user" and typ != "function_call_output":
            flush_assistant()
            msgs.append({"role": "user", "content": _text_of(it) or it.get("content", "")})
        elif typ == "message":
            # a new reasoning block starts a fresh assistant turn if calls already queued
            if pending_calls:
                flush_assistant()
            pending_text += _text_of(it)
        elif typ == "function_call":
            args = it.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            pending_calls.append({"type": "function",
                                  "function": {"name": it.get("name", "bash"),
                                               "arguments": args}})
        elif typ == "function_call_output":
            flush_assistant()
            out = it.get("output")
            msgs.append({"role": "tool",
                         "content": out if isinstance(out, str) else json.dumps(out)})
    flush_assistant()
    return msgs


def resolved_instance_ids(results_csv: Path) -> set[str]:
    ids: set[str] = set()
    with results_csv.open() as f:
        for row in csv.DictReader(f):
            if str(row.get("resolved", "")).strip().lower() == "true":
                ids.add(row["instance_id"])
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path, required=True, action="append",
                    help="a <model> trajectory dir (repeatable to pool several runs)")
    ap.add_argument("--results", type=Path, required=True, action="append",
                    help="the results.csv paired with each --traj-dir (same order); "
                         "used to keep RESOLVED instances only")
    ap.add_argument("--all-patched", action="store_true",
                    help="train on all trajectories with a non-empty patch, not just "
                         "resolved ones (reintroduces the mimicry risk — off by default)")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--max-tool-tokens", type=int, default=1024,
                    help="head+tail truncate tool outputs (container dumps are huge; "
                         "raises trainable density)")
    ap.add_argument("--min-density", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=REPO / "out" / "exp-058" / "distill_corpus.pt")
    args = ap.parse_args()

    if len(args.traj_dir) != len(args.results):
        sys.exit("[distill] --traj-dir and --results must be given the same number of times")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.chat_template = CHAT_TEMPLATE.read_text()

    ids_stream: list[int] = []
    lbl_stream: list[int] = []
    tot_asst = 0
    n_kept = 0
    n_skipped = 0
    n_trunc_msgs = 0
    trunc_saved = 0
    per_source: list[str] = []

    for traj_dir, results in zip(args.traj_dir, args.results):
        keep = None if args.all_patched else resolved_instance_ids(results)
        if keep is not None:
            print(f"[distill] {results.name}: {len(keep)} resolved instances", flush=True)
        traj_files = sorted(traj_dir.glob("*.traj.json"))
        kept_here = 0
        for tf in traj_files:
            inst = tf.name[: -len(".traj.json")]
            if keep is not None and inst not in keep:
                n_skipped += 1
                continue
            blob = json.loads(tf.read_text())
            msgs = trajectory_to_messages(blob["messages"])
            # need at least one assistant turn to train on
            if not any(m["role"] == "assistant" for m in msgs):
                n_skipped += 1
                continue
            msgs, n_tr, saved = truncate_tool_messages(msgs, tok, args.max_tool_tokens)
            n_trunc_msgs += n_tr
            trunc_saved += saved
            ids, lbl = masked_ids_for_session(msgs, tok, tools=[BASH_TOOL])
            ids_stream += ids
            lbl_stream += lbl
            tot_asst += sum(1 for x in lbl if x != -100)
            n_kept += 1
            kept_here += 1
        per_source.append(f"{traj_dir.parent.parent.name}:{kept_here}")
        print(f"[distill]   {traj_dir}: kept {kept_here}/{len(traj_files)} trajectories",
              flush=True)

    if not ids_stream:
        sys.exit("[distill] no trajectories kept — check --results resolved flags / --traj-dir")

    windows = pack(ids_stream, lbl_stream, args.window, min_density=args.min_density)
    if not windows:
        sys.exit("[distill] no windows survived packing — lower --min-density or check density")
    frac = 100 * tot_asst / max(1, len(ids_stream))
    print(f"[distill] {n_kept} trajectories ({n_skipped} skipped) -> {len(ids_stream):,} tokens "
          f"({frac:.0f}% assistant-masked) -> {len(windows)} windows of {args.window}", flush=True)
    if args.max_tool_tokens > 0:
        print(f"[distill] tool-output truncation: {n_trunc_msgs} messages, "
              f"{trunc_saved:,} masked tokens dropped", flush=True)

    import random
    rng = random.Random(42)
    rng.shuffle(windows)
    ids_t = torch.tensor([w["ids"] for w in windows], dtype=torch.long)
    lbl_t = torch.tensor([w["labels"] for w in windows], dtype=torch.long)

    # same density-audit + stop-token tripwire as the log-corpus builder
    dens = (lbl_t != -100).float().mean(dim=1)
    deciles = torch.quantile(dens, torch.linspace(0, 1, 11))
    print("[distill] window trainable-density deciles: "
          + " ".join(f"{d:.2f}" for d in deciles.tolist()), flush=True)
    im_end_id = tok.convert_tokens_to_ids(IM_END)
    n_im_end = (0 if im_end_id is None
                else int(((lbl_t == im_end_id) & (lbl_t != -100)).sum()))
    print(f"[distill] labeled {IM_END} targets: {n_im_end}", flush=True)
    assert n_im_end > 0, "no labeled stop-token targets — masking regressed"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fp = corpus_fingerprint(ids_t, lbl_t)
    torch.save({"ids": ids_t, "labels": lbl_t, "window": args.window,
                "tool_windows": len(windows), "wiki_windows": 0,
                "assistant_frac": frac / 100, "split": "distill-swebench",
                "min_density": args.min_density, "max_tool_tokens": args.max_tool_tokens,
                "im_end_targets": n_im_end, "n_trajectories": n_kept,
                "sources": per_source, "resolved_only": not args.all_patched,
                "fingerprint": fp}, args.out)
    print(f"[distill] saved {ids_t.shape[0]} windows [{tuple(ids_t.shape)}] "
          f"fingerprint={fp} -> {args.out}")
    print(f"[distill] masked-token share overall: {100*(lbl_t!=-100).float().mean():.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
