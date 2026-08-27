#!/usr/bin/env python3
"""P(<|im_end|>) at fixed probe points — the termination-policy endpoint.

Continued QAT on the ternary model can leave grammar perfectly intact while moving the
*stopping policy*: `sft32k` (--stop-weight 6.0) writes one correct sentence and then emits
`<|im_end|>` instead of the tool call. Loss and perplexity cannot see that. This measures it
directly: post a raw templated prompt to llama.cpp `/completion` with `n_predict=1,
n_probs=N, temperature=0` and read the probability and rank of the stop token in
`completion_probabilities[0].top_logprobs`.

The probe points walk the positions where a stop decision is plausible, from "must not stop"
to "may legitimately stop":

    start            generation has just opened; stopping here means emitting nothing
    mid_sentence     mid-clause; stopping is a grammar error, not a policy choice
    sentence_period  a complete sentence + '.' — the absorbing state sft32k learned
    sentence_newline the same, plus a newline
    after_tool_call  a complete <tool_call> block — here a stop IS defensible

Read `sentence_period` against `start`: a model that terminates correctly keeps both low
during an agentic turn, because the turn is supposed to continue into a tool call.

Prompts are built through the server's own `/apply-template`, so they are exactly what the
model's chat template produces — never hand-assembled control tokens.

Usage
-----
    # against an already-running server
    python scripts/probe_stop_prob.py --base-url http://127.0.0.1:18081 --label vanilla

    # spawn one (CPU by default, so it can run beside a training job)
    python scripts/probe_stop_prob.py --model out/exp-057/...-Q2_0.gguf --label sft32k_sw1

Q2_0 needs the prism fork: LLAMA_CPP_DIR=vendor/llama.cpp-prism.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# The Qwen-family stop token. Resolved from the server at runtime; this is the fallback
# and the value the sft32k measurements in docs/next_session_sft32k_sw1.md were read at.
DEFAULT_STOP_ID = 151645
STOP_PIECE = "<|im_end|>"

# One agentic turn, mirroring the SWE setting these runs are trained and graded on: a repo
# task with a bash tool in scope, where the correct continuation is a tool call, never a stop.
SYSTEM = "You are a coding agent working in a git repository."
USER = "The dask DataFrame shuffle fails on empty partitions. Find and fix the bug."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command"}
                },
                "required": ["command"],
            },
        },
    }
]

# The sentence is the model's own observed output under sft32k, kept verbatim so the
# probe lands on the exact boundary where that run emitted <|im_end|>.
SENTENCE = "Let me explore the repository structure and understand the bug."
TOOL_CALL = (
    '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>'
)

# (name, text appended to the generation prefix)
PROBE_POINTS: list[tuple[str, str]] = [
    ("start", ""),
    ("mid_sentence", "Let me explore the repository"),
    ("sentence_period", SENTENCE),
    ("sentence_newline", SENTENCE + "\n"),
    ("after_tool_call", SENTENCE + "\n" + TOOL_CALL),
]


def _post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode())


def wait_healthy(base_url: str, proc: subprocess.Popen | None, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {proc.returncode} before becoming healthy"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as fh:
                if fh.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2.0)
    raise RuntimeError(f"llama-server not healthy within {timeout:.0f}s")


def resolve_stop_id(base_url: str) -> int:
    """Ask the server for the id of the stop piece rather than trusting a constant."""
    try:
        out = _post(f"{base_url}/tokenize", {"content": STOP_PIECE, "special": True})
        toks = out.get("tokens") or []
        if len(toks) == 1:
            tok = toks[0]
            tid = tok["id"] if isinstance(tok, dict) else int(tok)
            return int(tid)
        print(
            f"[probe] warning: {STOP_PIECE!r} tokenized to {len(toks)} ids, not 1; "
            f"falling back to {DEFAULT_STOP_ID}",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"[probe] warning: /tokenize failed ({exc}); using {DEFAULT_STOP_ID}",
              file=sys.stderr)
    return DEFAULT_STOP_ID


def build_prefix(base_url: str) -> str:
    """Render the generation prefix with the model's own chat template."""
    out = _post(
        f"{base_url}/apply-template",
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
            "tools": TOOLS,
        },
    )
    prompt = out.get("prompt")
    if not prompt:
        raise RuntimeError(f"/apply-template returned no prompt: {out!r}")
    return prompt


def _as_prob(entry: dict) -> float:
    """top_logprobs entries carry `prob` on some builds and `logprob` on others."""
    if "prob" in entry and entry["prob"] is not None:
        return float(entry["prob"])
    if "logprob" in entry and entry["logprob"] is not None:
        return math.exp(float(entry["logprob"]))
    raise KeyError(f"no prob/logprob in {entry!r}")


def probe_one(base_url: str, prompt: str, stop_id: int, n_probs: int) -> dict:
    out = _post(
        f"{base_url}/completion",
        {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": n_probs,
            "temperature": 0.0,
            "cache_prompt": False,
        },
    )
    probs = out.get("completion_probabilities") or []
    if not probs:
        raise RuntimeError(
            "no completion_probabilities returned — does this build support n_probs?"
        )
    tops = probs[0].get("top_logprobs") or probs[0].get("probs") or []
    if not tops:
        raise RuntimeError(f"no top_logprobs in {probs[0]!r}")

    ranked = sorted(tops, key=_as_prob, reverse=True)
    stop_prob: float | None = None
    stop_rank: int | None = None
    for i, entry in enumerate(ranked, start=1):
        if int(entry.get("id", -1)) == stop_id:
            stop_prob = _as_prob(entry)
            stop_rank = i
            break

    top1 = ranked[0]
    return {
        "stop_prob": stop_prob,
        "stop_rank": stop_rank,
        "n_returned": len(ranked),
        # When the stop token misses the window entirely, the tail bound is the most
        # informative thing we can honestly report.
        "tail_bound": _as_prob(ranked[-1]),
        "top1_token": top1.get("token"),
        "top1_prob": _as_prob(top1),
        "sampled": probs[0].get("token") or out.get("content"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", type=Path, help="GGUF to serve (spawns llama-server)")
    src.add_argument("--base-url", help="already-running OpenAI-compatible llama-server")
    ap.add_argument("--label", required=True, help="name for this model in the output")
    ap.add_argument("--out", type=Path, help="append results as CSV here")
    ap.add_argument("--json-out", type=Path, help="write the full result blob here")
    ap.add_argument("--n-probs", type=int, default=60)
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--ngl", type=int, default=0,
                    help="GPU layers; 0 (default) keeps the card free for training")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--ctx-size", type=int, default=8192)
    ap.add_argument("--server-timeout", type=float, default=900.0)
    args = ap.parse_args()

    proc: subprocess.Popen | None = None
    if args.model:
        if not args.model.exists():
            print(f"missing GGUF: {args.model}", file=sys.stderr)
            return 1
        llama_dir = os.environ.get("LLAMA_CPP_DIR", "vendor/llama.cpp-prism")
        binary = Path(llama_dir) / "build" / "bin" / "llama-server"
        if not binary.exists():
            print(f"missing llama-server at {binary}", file=sys.stderr)
            return 1
        base_url = f"http://127.0.0.1:{args.port}"
        cmd = [
            str(binary), "--model", str(args.model),
            "--ctx-size", str(args.ctx_size),
            "--n-gpu-layers", str(args.ngl),
            "--threads", str(args.threads),
            "--jinja",  # without this tool calls are never parsed
            "--host", "127.0.0.1", "--port", str(args.port),
        ]
        print(f"[probe] spawning: {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        base_url = args.base_url.rstrip("/")

    try:
        wait_healthy(base_url, proc, args.server_timeout)
        stop_id = resolve_stop_id(base_url)
        prefix = build_prefix(base_url)
        print(f"[probe] {args.label}: stop id {stop_id}, prefix {len(prefix)} chars",
              file=sys.stderr)

        rows = []
        for name, suffix in PROBE_POINTS:
            res = probe_one(base_url, prefix + suffix, stop_id, args.n_probs)
            res.update({"label": args.label, "probe": name, "stop_id": stop_id})
            rows.append(res)

            if res["stop_prob"] is None:
                shown = f"< {res['tail_bound']:.2e} (outside top {res['n_returned']})"
                rank = f">{res['n_returned']}"
            else:
                shown = f"{res['stop_prob']:.7f}"
                rank = str(res["stop_rank"])
            print(f"  {name:<17} P(stop)={shown:<28} rank={rank:<5} "
                  f"top1={res['top1_token']!r} ({res['top1_prob']:.3f})")
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["label", "probe", "stop_id", "stop_prob", "stop_rank", "n_returned",
                  "tail_bound", "top1_token", "top1_prob", "sampled"]
        exists = args.out.exists()
        with args.out.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if not exists:
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})
        print(f"[probe] appended {len(rows)} rows -> {args.out}", file=sys.stderr)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
