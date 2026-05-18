#!/usr/bin/env python3
"""Tool-call accuracy benchmark for a single GGUF quant.

Runs the model against artifacts/toolcall_holdout.jsonl in two passes:

  1. Per-turn pass — for each ground-truth assistant tool-call turn, replay the
     prior context and score: tool selection, parameter accuracy, schema
     validity.
  2. Rollout pass — once per session, run the full agentic loop feeding in
     recorded tool results, and record whether the model sustains the
     conversation.

Outputs:
  experiments/_shared/toolcall_results.csv  — appended summary row
  experiments/_shared/logs/toolcall_<stem>_<ts>.jsonl  — per-turn details

Either point at a running llama-server (--base-url) or let the script spawn
one for you (default).
"""

import argparse
import csv
import json
import os
import random
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

import requests
from openai import OpenAI

try:
    from jsonschema import Draft7Validator
    _HAVE_JSONSCHEMA = True
except ImportError:
    _HAVE_JSONSCHEMA = False


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_DEFAULT_HOLDOUT = os.path.join(_ROOT, "out", "omnicoder_q4_k_m", "toolcall_holdout.jsonl")
_DEFAULT_OUT = os.path.join(_ROOT, "out", "omnicoder_q4_k_m", "toolcall_results.csv")
_LOG_DIR = os.path.join(_ROOT, "out", "omnicoder_q4_k_m", "logs")
_LLAMA_SERVER = os.path.join(_ROOT, "vendor", "llama.cpp", "build", "bin", "llama-server")


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_health(base_url: str, timeout: float = 60.0) -> None:
    url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise TimeoutError(f"llama-server at {url} did not become healthy in {timeout}s ({last_err})")


def spawn_server(model_path: str, ctx: int, ngl: int, port: int) -> subprocess.Popen:
    if not os.path.exists(_LLAMA_SERVER):
        raise FileNotFoundError(
            f"llama-server not found at {_LLAMA_SERVER}. "
            "Build it with `cmake --build llama.cpp/build --target llama-server` "
            "or pass --base-url to point at an existing server."
        )
    cmd = [
        _LLAMA_SERVER,
        "-m", model_path,
        "--jinja",
        "--port", str(port),
        "-c", str(ctx),
        "-ngl", str(ngl),
        "--host", "127.0.0.1",
    ]
    print(f"==> spawning: {' '.join(cmd)}", flush=True)
    log_path = os.path.join(_LOG_DIR, f"server_{os.path.basename(model_path)}_{int(time.time())}.log")
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    print(f"    server log: {log_path}", flush=True)
    return proc


# ---------------------------------------------------------------------------
# Argument / schema comparison
# ---------------------------------------------------------------------------


def parse_arguments(args: Any) -> dict | None:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            v = json.loads(args)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# Argument keys that hold free-form natural language or open-ended code/shell
# strings. Two reasonable agents will rarely emit byte-identical values for
# these, so we score them as presence-only when present.
_FREE_TEXT_ARG_KEYS = {
    "command", "description", "content", "message", "text", "prompt",
    "question", "questions", "new_string", "old_string", "code", "body",
    "instructions", "explanation", "summary", "thought", "thinking",
    "query", "search",
}


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


# Argument keys that hold filesystem paths. Compared with path normalization,
# case-insensitive fallback, and basename overlap.
_PATH_ARG_KEYS = {
    "file_path", "filepath", "filename", "path", "dir", "directory",
    "target_file", "source", "destination",
}

# Argument keys that hold shell commands. Compared with shlex parsing.
_COMMAND_ARG_KEYS = {"command", "cmd", "shell_command"}


def _norm_path(s: str) -> str:
    return os.path.normpath(os.path.expanduser(str(s))).rstrip("/")


def _compare_path(a: Any, b: Any) -> tuple[bool, bool]:
    if not (isinstance(a, str) and isinstance(b, str)):
        return False, False
    na, nb = _norm_path(a), _norm_path(b)
    if na == nb:
        return True, True
    if na.lower() == nb.lower():
        return False, True  # case-insensitive match (HFS+, NTFS)
    if os.path.basename(na) and os.path.basename(na) == os.path.basename(nb):
        return False, True  # same file, different parent
    return False, False


def _compare_number(a: Any, b: Any, tol: float = 0.10) -> tuple[bool, bool]:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False, False
    if fa == fb:
        return True, True
    denom = max(abs(fa), abs(fb), 1.0)
    return False, (abs(fa - fb) / denom) <= tol


def _compare_command(a: Any, b: Any) -> tuple[bool, bool]:
    if not (isinstance(a, str) and isinstance(b, str)):
        return False, False
    if a.strip() == b.strip():
        return True, True
    try:
        ta = shlex.split(a)
        tb = shlex.split(b)
    except ValueError:
        return False, False
    if not ta or not tb:
        return False, False
    if ta[0] != tb[0]:
        return False, False  # different base program → not similar
    sa, sb = set(ta[1:]), set(tb[1:])
    if not sa and not sb:
        return False, True  # both bare commands like `ls`
    inter = len(sa & sb)
    union = len(sa | sb)
    return False, (inter / union) >= 0.3 if union else False


def _compare_generic_string(a: Any, b: Any) -> tuple[bool, bool]:
    if a == b:
        return True, True
    if not (isinstance(a, str) and isinstance(b, str)):
        return False, False
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return False, True
    if not ta or not tb:
        return False, False
    return False, (len(ta & tb) / len(ta | tb)) >= 0.5


def compare_value(key: str, pred: Any, truth: Any, key_schema: dict | None) -> tuple[bool, bool, str]:
    """Type-aware comparison. Returns (exact, similar, method_used)."""
    k = key.lower()
    sch_type = (key_schema or {}).get("type") if key_schema else None

    if sch_type == "boolean" or isinstance(truth, bool):
        exact = pred == truth
        return exact, exact, "boolean"
    if sch_type in ("number", "integer") or isinstance(truth, (int, float)):
        e, s = _compare_number(pred, truth)
        return e, s, "numeric"
    if k in _PATH_ARG_KEYS or "path" in k:
        e, s = _compare_path(pred, truth)
        return e, s, "path"
    if k in _COMMAND_ARG_KEYS:
        e, s = _compare_command(pred, truth)
        return e, s, "command"
    if isinstance(truth, (list, dict)):
        eq = json.dumps(pred, sort_keys=True) == json.dumps(truth, sort_keys=True)
        return eq, eq, "structural"
    e, s = _compare_generic_string(pred, truth)
    return e, s, "string-jaccard"


# Backwards-compat shim — anything still calling value_similar gets generic.
def value_similar(a: Any, b: Any) -> tuple[bool, bool]:
    return _compare_generic_string(a, b)


def schema_for(tool_name: str, tools: list[dict]) -> dict | None:
    for t in tools:
        fn = t.get("function") or {}
        if fn.get("name") == tool_name or t.get("name") == tool_name:
            return fn.get("parameters") or t.get("parameters")
    return None


def is_schema_valid(tool_name: str, args: dict | None, tools: list[dict]) -> tuple[bool, str]:
    if not any((t.get("function") or {}).get("name") == tool_name or t.get("name") == tool_name for t in tools):
        return False, f"tool '{tool_name}' not in tools list"
    if args is None:
        return False, "arguments did not parse as JSON object"
    sch = schema_for(tool_name, tools)
    if not sch:
        return True, "no schema; presence-only check"
    if _HAVE_JSONSCHEMA:
        errs = sorted(Draft7Validator(sch).iter_errors(args), key=lambda e: e.path)
        if errs:
            return False, "; ".join(f"{list(e.path)}: {e.message}" for e in errs[:3])
        return True, "ok"
    # Fallback: required keys + crude type check.
    missing = [k for k in (sch.get("required") or []) if k not in args]
    if missing:
        return False, f"missing required: {missing}"
    return True, "ok (no jsonschema lib)"


def param_score(pred_args: dict | None, truth_args: dict, schema: dict | None) -> tuple[float, dict]:
    """Fraction of required keys present and value-equivalent (similar)."""
    if pred_args is None:
        return 0.0, {"missing_all": True}
    required = list((schema or {}).get("required") or list(truth_args.keys()))
    if not required:
        return 1.0, {"required": []}
    props = (schema or {}).get("properties") or {}
    hits = 0
    details: dict[str, Any] = {}
    for k in required:
        if k not in pred_args:
            details[k] = "missing"
            continue
        if k.lower() in _FREE_TEXT_ARG_KEYS:
            details[k] = "present (free-text)"
            hits += 1
            continue
        if k not in truth_args:
            details[k] = "present (no truth)"
            hits += 1
            continue
        exact, similar, method = compare_value(k, pred_args[k], truth_args[k], props.get(k))
        if exact:
            details[k] = f"exact ({method})"
            hits += 1
        elif similar:
            details[k] = f"similar ({method})"
            hits += 1
        else:
            details[k] = {
                "result": f"mismatch ({method})",
                "pred": _shortrepr(pred_args[k]),
                "truth": _shortrepr(truth_args[k]),
            }
    return hits / len(required), details


def _shortrepr(v: Any, maxlen: int = 120) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= maxlen else s[:maxlen] + "..."


# ---------------------------------------------------------------------------
# Eval loops
# ---------------------------------------------------------------------------


DEFAULT_SYSTEM_PROMPT = (
    "You are a coding assistant operating in an agentic loop. "
    "Use the provided tools to complete the user's task. "
    "Prefer calling a tool over describing what you would do."
)


def maybe_inject_system(msgs: list[dict], system_prompt: str | None) -> list[dict]:
    """If the session has no system message, prepend the provided prompt.
    Returns a new list; does not mutate input."""
    if not system_prompt:
        return msgs
    if any(m.get("role") == "system" for m in msgs):
        return msgs
    return [{"role": "system", "content": system_prompt}, *msgs]


def strip_for_api(msgs: list[dict]) -> list[dict]:
    """Strip non-OpenAI fields and reshape to chat/completions format."""
    out = []
    for m in msgs:
        role = m.get("role")
        if role == "user":
            out.append({"role": "user", "content": m.get("content", "") or ""})
        elif role == "system":
            out.append({"role": "system", "content": m.get("content", "") or ""})
        elif role == "assistant":
            o: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                tcs = []
                for tc in m["tool_calls"]:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    if not isinstance(args, str):
                        args = json.dumps(args or {})
                    tcs.append({
                        "id": tc.get("id") or f"call_{len(tcs)}",
                        "type": "function",
                        "function": {"name": fn.get("name"), "arguments": args},
                    })
                o["tool_calls"] = tcs
            out.append(o)
        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id") or "call_0",
                "content": m.get("content", "") or "",
            })
    return out


def call_model(client: OpenAI, messages: list[dict], tools: list[dict],
               temperature: float, max_tokens: int) -> Any:
    return client.chat.completions.create(
        model="local",
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def eval_per_turn(client: OpenAI, session: dict, temperature: float, max_tokens: int,
                  max_turns: int, log_fh, system_prompt: str | None = None,
                  stop_on_fail: bool = True) -> list[dict]:
    """Score every assistant tool_call turn in the session."""
    results = []
    msgs = maybe_inject_system(session["messages"], system_prompt)
    tools = session["tools"]
    sid = session.get("session_id")
    src = session.get("source")

    turn_idx = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        if turn_idx >= max_turns:
            break
        turn_idx += 1

        prefix = strip_for_api(msgs[:i])
        truth_tc = m["tool_calls"][0]
        truth_name = (truth_tc.get("function") or {}).get("name") or truth_tc.get("name")
        truth_args = parse_arguments((truth_tc.get("function") or {}).get("arguments")
                                     or truth_tc.get("arguments")) or {}

        try:
            resp = call_model(client, prefix, tools, temperature, max_tokens)
            choice = resp.choices[0].message
            pred_tcs = choice.tool_calls or []
        except Exception as e:
            rec = {"session": sid, "source": src, "turn": turn_idx, "error": str(e),
                   "truth_name": truth_name, "selection": False,
                   "param_acc": 0.0, "schema_valid": False}
            log_fh.write(json.dumps(rec) + "\n")
            results.append(rec)
            if stop_on_fail:
                break
            continue

        if not pred_tcs:
            rec = {"session": sid, "source": src, "turn": turn_idx, "truth_name": truth_name,
                   "pred_name": None, "selection": False, "param_acc": 0.0,
                   "schema_valid": False, "note": "no tool_calls emitted",
                   "pred_content": (choice.content or "")[:200]}
            log_fh.write(json.dumps(rec) + "\n")
            results.append(rec)
            if stop_on_fail:
                break
            continue

        pred_tc = pred_tcs[0]
        pred_name = pred_tc.function.name
        pred_args = parse_arguments(pred_tc.function.arguments)

        selection = (pred_name or "").lower() == (truth_name or "").lower()
        sch = schema_for(truth_name, tools)
        pacc, pdetails = param_score(pred_args, truth_args, sch) if selection else (0.0, {"wrong_tool": True})
        schema_ok, schema_msg = is_schema_valid(pred_name, pred_args, tools)

        rec = {
            "session": sid, "source": src, "turn": turn_idx,
            "truth_name": truth_name, "pred_name": pred_name,
            "selection": selection, "param_acc": pacc, "param_details": pdetails,
            "schema_valid": schema_ok, "schema_msg": schema_msg,
            "truth_args_keys": sorted(truth_args.keys()),
            "pred_args_keys": sorted((pred_args or {}).keys()),
        }
        log_fh.write(json.dumps(rec) + "\n")
        results.append(rec)
        if stop_on_fail and not selection:
            break
    return results


def eval_rollout(client: OpenAI, session: dict, temperature: float, max_tokens: int,
                 max_turns: int, log_fh, system_prompt: str | None = None) -> dict:
    """Run a single rollout, splicing in recorded tool results by call order per tool."""
    msgs = maybe_inject_system(session["messages"], system_prompt)
    tools = session["tools"]
    sid = session.get("session_id")

    # Index recorded tool results by tool name, in order.
    truth_results_by_tool: dict[str, list[str]] = {}
    for m in msgs:
        if m.get("role") == "tool":
            name = m.get("name")
            if name:
                truth_results_by_tool.setdefault(name, []).append(m.get("content", ""))

    # Start with system + first user turn(s) before any assistant turn.
    convo: list[dict] = []
    for m in msgs:
        if m.get("role") in ("system", "user"):
            convo.append({"role": m["role"], "content": m.get("content", "") or ""})
        else:
            break
    if not convo:
        return {"session": sid, "completed": False, "reason": "no user turn"}

    consumed: dict[str, int] = {}
    tools_called: list[str] = []
    completed_reason = "max_turns"
    completed = False

    for turn in range(max_turns):
        try:
            resp = call_model(client, convo, tools, temperature, max_tokens)
            choice = resp.choices[0].message
        except Exception as e:
            completed_reason = f"error: {e}"
            break

        tcs = choice.tool_calls or []
        if not tcs:
            completed = True
            completed_reason = "model stopped"
            convo.append({"role": "assistant", "content": choice.content or ""})
            break

        # Append model's assistant turn with tool_calls.
        api_tcs = []
        for tc in tcs:
            api_tcs.append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })
        convo.append({"role": "assistant", "content": choice.content or None, "tool_calls": api_tcs})

        for tc in tcs:
            name = tc.function.name
            tools_called.append(name)
            idx = consumed.get(name, 0)
            results_for = truth_results_by_tool.get(name) or []
            if idx < len(results_for):
                content = results_for[idx]
            else:
                content = f"(no recorded result for {name} call #{idx + 1})"
            consumed[name] = idx + 1
            convo.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    truth_set = set((session.get("tools_used") or []))
    pred_set = {n.lower() for n in tools_called}
    truth_set_l = {n.lower() for n in truth_set}
    tool_set_match = bool(truth_set_l) and pred_set == truth_set_l

    rec = {
        "session": sid, "source": session.get("source"),
        "completed": completed, "reason": completed_reason,
        "n_tool_calls": len(tools_called), "tools_called": tools_called,
        "truth_tools_used": list(truth_set), "tool_set_match": tool_set_match,
    }
    log_fh.write(json.dumps({"rollout": rec}) + "\n")
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="Path to GGUF (required unless --base-url given without spawning)")
    p.add_argument("--base-url", help="If set, skip server spawn and use this OpenAI-compatible base URL")
    p.add_argument("--holdout", default=_DEFAULT_HOLDOUT)
    p.add_argument("--out", default=_DEFAULT_OUT)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--max-turns-per-session", type=int, default=8,
                   help="Cap on per-turn evaluations per session (keeps wall time bounded)")
    p.add_argument("--rollout-max-turns", type=int, default=12)
    p.add_argument("--skip-rollout", action="store_true")
    p.add_argument("--server-startup-timeout", type=float, default=120.0)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                   help="System prompt to inject when a session lacks one. "
                        "Pass empty string to disable injection.")
    p.add_argument("--no-stop-on-fail", action="store_true",
                   help="Continue scoring further turns even after the model picks "
                        "the wrong tool (or no tool). Default is to stop the session "
                        "at the first selection failure.")
    args = p.parse_args()

    if not os.path.exists(args.holdout):
        print(f"ERROR: holdout not found at {args.holdout}. Run build_toolcall_holdout.py first.",
              file=sys.stderr)
        return 1

    if args.base_url:
        base_url = args.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        server_proc = None
        model_label = args.model or "remote"
    else:
        if not args.model:
            print("ERROR: --model is required when not using --base-url", file=sys.stderr)
            return 1
        if not os.path.exists(args.model):
            print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
            return 1
        port = free_port()
        server_proc = spawn_server(args.model, args.ctx, args.ngl, port)
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            wait_for_health(base_url, timeout=args.server_startup_timeout)
        except Exception as e:
            server_proc.send_signal(signal.SIGTERM)
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        model_label = args.model

    client = OpenAI(base_url=base_url, api_key="sk-no-key")

    with open(args.holdout) as f:
        sessions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(sessions)} holdout sessions from {args.holdout}")

    os.makedirs(_LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(model_label))[0]
    log_path = os.path.join(_LOG_DIR, f"toolcall_{stem}_{ts}.jsonl")
    log_fh = open(log_path, "w")

    all_turn_results: list[dict] = []
    all_rollouts: list[dict] = []

    try:
        for si, sess in enumerate(sessions, 1):
            print(f"[{si}/{len(sessions)}] {sess.get('session_id')}")
            turns = eval_per_turn(client, sess, args.temperature, args.max_tokens,
                                  args.max_turns_per_session, log_fh,
                                  system_prompt=args.system_prompt or None,
                                  stop_on_fail=not args.no_stop_on_fail)
            all_turn_results.extend(turns)
            sel = sum(1 for t in turns if t.get("selection"))
            print(f"    per-turn: {len(turns)} eval'd, selection {sel}/{len(turns)}")

            if not args.skip_rollout:
                roll = eval_rollout(client, sess, args.temperature, args.max_tokens,
                                    args.rollout_max_turns, log_fh,
                                    system_prompt=args.system_prompt or None)
                all_rollouts.append(roll)
                print(f"    rollout: completed={roll['completed']} reason={roll['reason']}"
                      f" tool_set_match={roll['tool_set_match']}")
    finally:
        log_fh.close()
        if server_proc is not None:
            print("==> terminating server")
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    # Aggregate.
    n_turns = len(all_turn_results)
    n_sel = sum(1 for t in all_turn_results if t.get("selection"))
    pacc_mean = (sum(t.get("param_acc", 0.0) for t in all_turn_results) / n_turns) if n_turns else 0.0
    n_schema_ok = sum(1 for t in all_turn_results if t.get("schema_valid"))
    n_roll_done = sum(1 for r in all_rollouts if r.get("completed"))
    n_roll_match = sum(1 for r in all_rollouts if r.get("tool_set_match"))

    summary = {
        "model": os.path.basename(model_label),
        "n_sessions": len(sessions),
        "n_turns": n_turns,
        "tool_selection_acc": n_sel / n_turns if n_turns else 0.0,
        "param_acc_mean": pacc_mean,
        "schema_valid_rate": n_schema_ok / n_turns if n_turns else 0.0,
        "rollout_complete_rate": n_roll_done / len(all_rollouts) if all_rollouts else 0.0,
        "rollout_tool_set_match_rate": n_roll_match / len(all_rollouts) if all_rollouts else 0.0,
        "temperature": args.temperature,
        "timestamp": ts,
    }

    print("\n" + "=" * 60)
    print(f"Model: {summary['model']}")
    print(f"  Tool selection accuracy:   {n_sel}/{n_turns}  ({summary['tool_selection_acc']:.3f})")
    print(f"  Param accuracy (mean):     {summary['param_acc_mean']:.3f}")
    print(f"  Schema-valid rate:         {n_schema_ok}/{n_turns}  ({summary['schema_valid_rate']:.3f})")
    if all_rollouts:
        print(f"  Rollouts completed:        {n_roll_done}/{len(all_rollouts)}")
        print(f"  Tool-set match (rollout):  {n_roll_match}/{len(all_rollouts)}")

    by_src: dict[str, list[dict]] = {}
    for t in all_turn_results:
        by_src.setdefault(t.get("source") or "unknown", []).append(t)
    if len(by_src) > 1:
        print("\n  Per-source breakdown:")
        for src in sorted(by_src.keys()):
            ts = by_src[src]
            sel = sum(1 for t in ts if t.get("selection"))
            pa = sum(t.get("param_acc", 0.0) for t in ts) / len(ts) if ts else 0.0
            sv = sum(1 for t in ts if t.get("schema_valid"))
            print(f"    {src:10s} n={len(ts):3d}  sel={sel}/{len(ts)} ({sel/len(ts):.2f})  "
                  f"param={pa:.2f}  schema={sv}/{len(ts)} ({sv/len(ts):.2f})")

    print(f"\n  Per-turn log:              {log_path}")

    # Append CSV row.
    fields = list(summary.keys())
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow(summary)
    print(f"  Appended to:               {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
