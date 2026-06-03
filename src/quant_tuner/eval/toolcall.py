"""Per-turn tool-call evaluation: replay each ground-truth turn and score it."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from openai import OpenAI, RateLimitError

from quant_tuner.eval.scoring import (
    canonical_tool,
    parse_arguments,
    score_turn,
)
from quant_tuner.eval.server import running_server

DEFAULT_SYSTEM_PROMPT = (
    "You are a coding assistant operating in an agentic loop. "
    "Use the provided tools to complete the user's task. "
    "Prefer calling a tool over describing what you would do."
)


# ---------------------------------------------------------------------------
# Message reshaping
# ---------------------------------------------------------------------------


def maybe_inject_system(msgs: list[dict], system_prompt: str | None) -> list[dict]:
    """Prepend ``system_prompt`` to ``msgs`` unless a system turn is already present.

    Returns a new list; does not mutate ``msgs``. Pass ``None`` or empty string
    to disable injection entirely.
    """
    if not system_prompt:
        return msgs
    if any(m.get("role") == "system" for m in msgs):
        return msgs
    return [{"role": "system", "content": system_prompt}, *msgs]


def strip_for_api(msgs: list[dict]) -> list[dict]:
    """Drop non-OpenAI fields and reshape to chat/completions message format.

    Two normalizations are applied so the result is valid for any
    OpenAI-compatible API (including those that reject assistant prefill):

    1. Consecutive assistant messages (e.g. a thinking turn followed by a
       tool-call turn) are merged into a single assistant message.
    2. A trailing assistant message that carries only text (no tool_calls)
       is dropped — it is a reasoning/thinking prefix that should not be
       sent as prefill since not all APIs support it.
    """
    out: list[dict] = []
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
            # Merge into previous assistant message if consecutive
            if out and out[-1]["role"] == "assistant":
                prev = out[-1]
                parts = [p for p in [prev["content"], o["content"]] if p]
                prev["content"] = "\n".join(parts) if parts else None
                if o.get("tool_calls"):
                    prev.setdefault("tool_calls", []).extend(o["tool_calls"])
            else:
                out.append(o)
        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id") or "call_0",
                "content": m.get("content", "") or "",
            })
    # Drop trailing content-only assistant message (thinking/reasoning prefix)
    # to avoid prefill errors on APIs that don't support it.
    while out and out[-1]["role"] == "assistant" and not out[-1].get("tool_calls"):
        out.pop()
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass
class Sampling:
    """Sampling params passed to ``/v1/chat/completions``.

    OpenAI-standard fields are forwarded directly; llama.cpp extensions
    (``top_k``, ``min_p``, ``repetition_penalty``) ride through ``extra_body``.
    """

    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None

    def to_request_kwargs(self) -> dict:
        kwargs: dict = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        extra: dict = {}
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.top_k is not None:
            extra["top_k"] = self.top_k
        if self.min_p is not None:
            extra["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            extra["repeat_penalty"] = self.repetition_penalty
        if extra:
            kwargs["extra_body"] = extra
        return kwargs


def call_model(
    client: OpenAI,
    messages: list[dict],
    tools: list[dict],
    sampling: Sampling,
    *,
    model_name: str = "local",
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> Any:
    """Call ``/v1/chat/completions`` with retry on rate-limit (429) errors.

    Retries up to *max_retries* times with exponential backoff starting at
    *base_delay* seconds (2 s → 4 s → 8 s → 16 s → 32 s).
    """
    # Omit ``tools`` entirely when empty: an empty array is rejected by strict
    # OpenAI servers (e.g. vLLM with --enable-auto-tool-choice). MMLU calls pass
    # tools=[]; tool-call calls pass a populated list.
    tools_kwarg = {"tools": tools} if tools else {}
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                **tools_kwarg,
                **sampling.to_request_kwargs(),
            )
        except RateLimitError:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Eval loops
# ---------------------------------------------------------------------------


# Anchored error signals. Unlike the old bare-substring heuristic (which fired
# on any file dump containing the word "error" — ~4940 false positives on the
# real corpus), these only match structured error markers: a line that *starts*
# with Error/Traceback, a non-zero exit code, or specific shell failures. This
# keeps the recovery signal meaningful instead of noise.
_ERROR_ANCHORS = (
    re.compile(r"^\s*error[:\s]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*traceback\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bexit code:?\s*[1-9]", re.IGNORECASE),
    re.compile(r"\bnon-zero exit\b", re.IGNORECASE),
    re.compile(r"\bcommand not found\b", re.IGNORECASE),
    re.compile(r"\bno such file or directory\b", re.IGNORECASE),
    re.compile(r"\bpermission denied\b", re.IGNORECASE),
)


def _content_looks_like_error(content: str | None) -> bool:
    """Tightened heuristic: does this tool-result string read like a real error?

    Matches only anchored signals (line-start ``Error:``/``Traceback``, non-zero
    exit code, or specific shell failures), never bare substrings. Used only as
    a fallback when a tool-result message has no authoritative ``is_error`` flag.
    """
    if not content:
        return False
    return any(rx.search(content) for rx in _ERROR_ANCHORS)


def tool_result_is_error(msg: dict) -> bool:
    """Did this ``tool``-role message report an error?

    Prefers the authoritative ``is_error`` flag when present (Claude Code logs
    carry it); otherwise falls back to the anchored content heuristic
    (:func:`_content_looks_like_error`) for sources that don't record the flag
    (qwen / opencode).
    """
    if "is_error" in msg:
        return bool(msg["is_error"])
    return _content_looks_like_error(msg.get("content"))


def prior_tool_result(msgs: list[dict], i: int) -> dict | None:
    """The message immediately before index ``i`` if it's a ``tool`` role, else None."""
    if i <= 0:
        return None
    prev = msgs[i - 1]
    return prev if prev.get("role") == "tool" else None


def _truth_calls_from_message(m: dict) -> tuple[list[tuple[str, dict]], bool]:
    """Parse ground-truth ``tool_calls`` from an assistant message.

    Returns ``(calls, malformed)``. ``malformed`` is True if any call's
    arguments failed to parse as a JSON object — those calls fall back to
    ``{}`` but the flag lets the orchestrator separate data-quality issues
    from model misses.
    """
    calls: list[tuple[str, dict]] = []
    malformed = False
    for tc in m.get("tool_calls") or []:
        name = (tc.get("function") or {}).get("name") or tc.get("name")
        raw = (tc.get("function") or {}).get("arguments") or tc.get("arguments")
        parsed = parse_arguments(raw)
        if parsed is None and raw not in (None, "", {}):
            malformed = True
        calls.append((name, parsed or {}))
    return calls, malformed


def _score_post_result(
    prior: dict,
    truth_msg: dict,
    pred_tcs: list,
    pred_content: str | None,
) -> dict:
    """Compare model's continuation against ground truth for an assistant turn
    that immediately follows a ``tool`` message.
    """
    truth_kind = "tool_call" if truth_msg.get("tool_calls") else "final_answer"
    pred_kind = "tool_call" if pred_tcs else "final_answer"
    rec: dict = {
        "truth_kind": truth_kind,
        "pred_kind": pred_kind,
        "continuation_type_match": truth_kind == pred_kind,
        "prior_was_error": tool_result_is_error(prior),
    }
    if not rec["prior_was_error"]:
        rec["recovery_appropriate"] = None
        return rec
    # Recovery heuristic: when the prior result was an error, an appropriate
    # next action either gives up (text reply) or retries with a *different*
    # tool name or different arguments than the call that produced the error.
    if pred_kind == "final_answer":
        rec["recovery_appropriate"] = True
        return rec
    prior_call_id = prior.get("tool_call_id")
    prior_call = _find_call_by_id(truth_msg, prior_call_id)  # may be None
    # Look further back: scan the conversation by tool_call_id is not enough;
    # the failing call is in an *earlier* assistant turn. We pass the failing
    # call's signature via prior.get("_failing_call") if available; otherwise
    # fall back to comparing tool name against any pred call.
    failing = prior.get("_failing_call") or prior_call
    if not failing:
        rec["recovery_appropriate"] = None
        return rec
    fname = failing.get("name")
    fargs = failing.get("args") or {}
    changed = any(
        p.function.name != fname
        or (parse_arguments(p.function.arguments) or {}) != fargs
        for p in pred_tcs
    )
    rec["recovery_appropriate"] = changed
    return rec


def _find_call_by_id(msg: dict, call_id: str | None) -> dict | None:
    if not call_id:
        return None
    for tc in msg.get("tool_calls") or []:
        if tc.get("id") == call_id:
            return {
                "name": (tc.get("function") or {}).get("name") or tc.get("name"),
                "args": parse_arguments(
                    (tc.get("function") or {}).get("arguments") or tc.get("arguments")
                ) or {},
            }
    return None


def _annotate_failing_calls(msgs: list[dict]) -> None:
    """Attach ``_failing_call`` to each ``tool`` message whose content looks
    like an error, pointing back to the assistant ``tool_calls`` entry that
    produced it (matched by ``tool_call_id``). Mutates ``msgs`` in place.
    """
    by_id: dict[str, dict] = {}
    for m in msgs:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                tid = tc.get("id")
                if tid:
                    by_id[tid] = {
                        "name": (tc.get("function") or {}).get("name") or tc.get("name"),
                        "args": parse_arguments(
                            (tc.get("function") or {}).get("arguments")
                            or tc.get("arguments")
                        ) or {},
                    }
        elif m.get("role") == "tool" and tool_result_is_error(m):
            failing = by_id.get(m.get("tool_call_id"))
            if failing is not None:
                m["_failing_call"] = failing


def eval_per_turn(
    client: OpenAI,
    session: dict,
    sampling: Sampling,
    *,
    max_turns: int,
    log_fh: IO[str] | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    stop_on_fail: bool = True,
    model_name: str = "local",
) -> list[dict]:
    """Score assistant turns in ``session``.

    Two kinds of records are emitted:

    * ``kind="tool_call"`` — an assistant turn that issues ``tool_calls``.
      Compared via :func:`score_turn` (selection / param_acc / schema_valid).
    * ``kind="post_result"`` — an assistant turn that immediately follows a
      ``tool`` message. Scores whether the model's continuation type matches
      ground truth and, when the prior tool result looks like an error,
      whether the model's next action is an appropriate recovery.

    A single assistant turn can produce both records (one model call, two
    score views). Turns where ground-truth ``tool_calls`` arguments fail to
    parse are still emitted but flagged ``truth_malformed=True`` so the
    aggregator can exclude them from denominators.
    """
    results: list[dict] = []
    msgs = maybe_inject_system(session["messages"], system_prompt)
    _annotate_failing_calls(msgs)
    tools = session["tools"]
    sid = session.get("session_id")
    src = session.get("source")
    stratum = session.get("stratum")
    n_truth_turns = sum(
        1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")
    )

    turn_idx = 0
    last_scored_truth = 0
    stopped_early = False
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        has_tool_calls = bool(m.get("tool_calls"))
        prior = prior_tool_result(msgs, i)
        if not has_tool_calls and prior is None:
            continue
        if turn_idx >= max_turns:
            break
        turn_idx += 1
        if has_tool_calls:
            last_scored_truth += 1

        base = {
            "session": sid, "source": src, "stratum": stratum, "turn": turn_idx,
        }
        prefix = strip_for_api(msgs[:i])

        try:
            resp = call_model(client, prefix, tools, sampling,
                              model_name=model_name)
            choice = resp.choices[0].message
            pred_tcs = choice.tool_calls or []
            pred_content = choice.content
        except Exception as e:
            rec = {
                **base, "kind": "tool_call" if has_tool_calls else "post_result",
                "error": str(e), "selection": False,
                "param_acc": 0.0, "schema_valid": False,
            }
            results.append(rec)
            if log_fh is not None:
                log_fh.write(json.dumps(rec) + "\n")
            if stop_on_fail:
                stopped_early = True
                break
            continue

        if has_tool_calls:
            truth_calls, truth_malformed = _truth_calls_from_message(m)
            truth_names = [n for n, _ in truth_calls]
            pred_calls: list[tuple[str, dict | None]] = [
                (p.function.name, parse_arguments(p.function.arguments)) for p in pred_tcs
            ]
            scored = score_turn(pred_calls, truth_calls, tools)
            rec = {
                **base, "kind": "tool_call",
                "truth_names": truth_names,
                "pred_names": [n for n, _ in pred_calls],
                "truth_malformed": truth_malformed,
                **scored,
            }
            if not pred_calls:
                rec["pred_content"] = (pred_content or "")[:200]
            results.append(rec)
            if log_fh is not None:
                log_fh.write(json.dumps(rec) + "\n")
            if stop_on_fail and not truth_malformed and not scored["selection"]:
                stopped_early = True
                break

        if prior is not None:
            post = _score_post_result(prior, m, pred_tcs, pred_content)
            rec = {**base, "kind": "post_result", **post}
            results.append(rec)
            if log_fh is not None:
                log_fh.write(json.dumps(rec) + "\n")

    # Attach session-level completion marker to the first record so the
    # aggregator can count completion states without re-walking sessions.
    if results:
        results[0]["_session_meta"] = {
            "n_truth_turns": n_truth_turns,
            "scored_truth_turns": last_scored_truth,
            "stopped_early": stopped_early,
        }
    return results


def eval_rollout(
    client: OpenAI,
    session: dict,
    sampling: Sampling,
    *,
    max_turns: int,
    log_fh: IO[str] | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    model_name: str = "local",
) -> dict:
    """Run a single rollout, splicing in recorded tool results by call order per tool.

    Currently unwired from :func:`run_toolcall_eval` — kept for future
    agentic-loop benchmarks. Per-turn replay is the canonical eval path.
    """
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

    for _ in range(max_turns):
        try:
            resp = call_model(client, convo, tools, sampling,
                              model_name=model_name)
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

        api_tcs = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tcs
        ]
        convo.append({"role": "assistant", "content": choice.content or None, "tool_calls": api_tcs})

        for tc in tcs:
            name = tc.function.name
            tools_called.append(name)
            idx = consumed.get(name, 0)
            results_for = truth_results_by_tool.get(name) or []
            content = (
                results_for[idx] if idx < len(results_for)
                else f"(no recorded result for {name} call #{idx + 1})"
            )
            consumed[name] = idx + 1
            convo.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    truth_set = set(session.get("tools_used") or [])
    pred_set = {n.lower() for n in tools_called}
    truth_set_l = {n.lower() for n in truth_set}
    tool_set_match = bool(truth_set_l) and pred_set == truth_set_l

    rec = {
        "session": sid, "source": session.get("source"),
        "completed": completed, "reason": completed_reason,
        "n_tool_calls": len(tools_called), "tools_called": tools_called,
        "truth_tools_used": list(truth_set), "tool_set_match": tool_set_match,
    }
    if log_fh is not None:
        log_fh.write(json.dumps({"rollout": rec}) + "\n")
    return rec


# ---------------------------------------------------------------------------
# Orchestrator: server lifecycle + both eval passes + summary
# ---------------------------------------------------------------------------


@dataclass
class EvalSummary:
    """Aggregate metrics returned by :func:`run_toolcall_eval`.

    Scalars are computed over ``tool_call``-kind turns with valid ground
    truth only (``truth_malformed`` and errored turns excluded from the
    denominator). ``n_scored`` is that denominator; ``n_total`` includes
    everything.

    ``param_acc_strict_mean`` is the exact-only/extra-key-penalized companion to
    ``param_acc_mean`` (see ``scoring.param_score_strict``); it is always
    ``<= param_acc_mean``. ``n_post_result_errors`` counts post-result turns
    whose prior tool result was a *real* error (``is_error`` flag when present,
    else an anchored heuristic) — not the old bare-substring count.
    """

    model: str
    n_sessions: int
    tool_selection_acc: float
    param_acc_mean: float
    schema_valid_rate: float
    param_acc_strict_mean: float = 0.0
    n_scored: int = 0
    n_total: int = 0
    continuation_type_match_rate: float = 0.0
    recovery_appropriate_rate: float = 0.0
    n_post_result: int = 0
    n_post_result_errors: int = 0
    per_turn: list[dict] = field(default_factory=list)
    by_source: dict[str, dict] = field(default_factory=dict)
    by_stratum: dict[str, dict] = field(default_factory=dict)
    per_tool: dict[str, dict] = field(default_factory=dict)
    by_turn_depth: dict[int, dict] = field(default_factory=dict)
    session_completion: dict[str, int] = field(default_factory=dict)
    truth_quality: dict[str, int] = field(default_factory=dict)


def _load_sessions(holdout: Path) -> list[dict]:
    with holdout.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _summarize_tool_call_group(ts: list[dict]) -> dict:
    n = len(ts)
    if not n:
        return {"n": 0, "tool_selection_acc": 0.0,
                "param_acc_mean": 0.0, "param_acc_strict_mean": 0.0,
                "schema_valid_rate": 0.0}
    return {
        "n": n,
        "tool_selection_acc": sum(1 for t in ts if t.get("selection")) / n,
        "param_acc_mean": sum(t.get("param_acc", 0.0) for t in ts) / n,
        "param_acc_strict_mean": sum(t.get("param_acc_strict", 0.0) for t in ts) / n,
        "schema_valid_rate": sum(1 for t in ts if t.get("schema_valid")) / n,
    }


def _aggregate(
    model_label: str,
    n_sessions: int,
    per_turn: list[dict],
) -> EvalSummary:
    # Session-completion accounting. Pre-pass: which sessions hit an error?
    sessions_with_error: set = set()
    for r in per_turn:
        if r.get("error"):
            sessions_with_error.add(r.get("session"))

    completion = {"complete": 0, "stopped_early": 0, "errored": 0, "empty": 0}
    for t in per_turn:
        meta = t.get("_session_meta")
        if not meta:
            continue
        sid = t.get("session")
        if sid in sessions_with_error:
            completion["errored"] += 1
        elif meta["stopped_early"]:
            completion["stopped_early"] += 1
        elif meta["scored_truth_turns"] == 0:
            completion["empty"] += 1
        else:
            completion["complete"] += 1

    # Tool-call-kind turns: split into scored vs. excluded (malformed truth
    # or errored API call) before computing the headline scalars.
    tc_turns = [t for t in per_turn if t.get("kind") == "tool_call"]
    n_total = len(tc_turns)
    scored = [
        t for t in tc_turns
        if not t.get("truth_malformed") and not t.get("error")
    ]
    n_scored = len(scored)
    truth_quality = {
        "malformed_truth": sum(1 for t in tc_turns if t.get("truth_malformed")),
        "api_errors": sum(1 for t in tc_turns if t.get("error")),
        "scored": n_scored,
        "total": n_total,
    }

    headline = _summarize_tool_call_group(scored)

    # Post-result kind: continuation-type + recovery rates.
    pr_turns = [t for t in per_turn if t.get("kind") == "post_result"]
    n_pr = len(pr_turns)
    n_pr_err = sum(1 for t in pr_turns if t.get("prior_was_error"))
    cont_rate = (
        sum(1 for t in pr_turns if t.get("continuation_type_match")) / n_pr
        if n_pr else 0.0
    )
    rec_subset = [
        t for t in pr_turns if t.get("recovery_appropriate") is not None
    ]
    rec_rate = (
        sum(1 for t in rec_subset if t.get("recovery_appropriate")) / len(rec_subset)
        if rec_subset else 0.0
    )

    # Stratified breakdowns over the scored tool_call turns.
    by_src: dict[str, list[dict]] = {}
    by_strat: dict[str, list[dict]] = {}
    by_depth: dict[int, list[dict]] = {}
    per_tool: dict[str, list[dict]] = {}
    for t in scored:
        by_src.setdefault(t.get("source") or "unknown", []).append(t)
        by_strat.setdefault(t.get("stratum") or "unknown", []).append(t)
        by_depth.setdefault(int(t.get("turn", 0)), []).append(t)
        # Per-tool: attribute the turn to every ground-truth tool name,
        # collapsed to its canonical capability so the same capability isn't
        # fragmented across per-agent spellings (read_file/Read/read → read).
        for name in t.get("truth_names") or []:
            per_tool.setdefault(canonical_tool(name), []).append(t)

    return EvalSummary(
        model=model_label,
        n_sessions=n_sessions,
        tool_selection_acc=headline["tool_selection_acc"],
        param_acc_mean=headline["param_acc_mean"],
        schema_valid_rate=headline["schema_valid_rate"],
        param_acc_strict_mean=headline["param_acc_strict_mean"],
        n_scored=n_scored,
        n_total=n_total,
        continuation_type_match_rate=cont_rate,
        recovery_appropriate_rate=rec_rate,
        n_post_result=n_pr,
        n_post_result_errors=n_pr_err,
        per_turn=per_turn,
        by_source={k: _summarize_tool_call_group(v) for k, v in by_src.items()},
        by_stratum={k: _summarize_tool_call_group(v) for k, v in by_strat.items()},
        per_tool={k: _summarize_tool_call_group(v) for k, v in per_tool.items()},
        by_turn_depth={k: _summarize_tool_call_group(v) for k, v in by_depth.items()},
        session_completion=completion,
        truth_quality=truth_quality,
    )


def run_toolcall_eval(
    holdout: Path,
    *,
    model_path: Path | None = None,
    base_url: str | None = None,
    sampling: Sampling | None = None,
    model_label: str | None = None,
    model_name: str = "local",
    max_turns_per_session: int = 8,
    stop_on_fail: bool = True,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    ctx: int = 8192,
    ngl: int = 99,
    server_log_path: Path | None = None,
    server_startup_timeout: float = 120.0,
    chat_template_kwargs: str | None = None,
    per_turn_log: Path | None = None,
    progress: bool = False,
    api_key: str = "sk-no-key",
) -> EvalSummary:
    """Run the per-turn pass against one model and return aggregates.

    Either ``model_path`` (spawn a server) **or** ``base_url`` (use a running
    one) must be provided. The two are mutually exclusive: pass ``model_path``
    for one-shot evals, ``base_url`` when an outer driver is reusing one server
    across multiple reps.
    """
    if (model_path is None) == (base_url is None):
        raise ValueError("provide exactly one of model_path or base_url")

    sampling = sampling or Sampling()
    sessions = _load_sessions(holdout)
    label = model_label or (model_path.name if model_path else "remote")

    log_fh = per_turn_log.open("w") if per_turn_log is not None else None

    def _run_against(url: str) -> EvalSummary:
        client = OpenAI(base_url=url, api_key=api_key)
        all_turns: list[dict] = []
        for i, sess in enumerate(sessions, 1):
            if progress:
                print(f"[{i}/{len(sessions)}] {sess.get('session_id')}", flush=True)
            turns = eval_per_turn(
                client, sess, sampling,
                max_turns=max_turns_per_session, log_fh=log_fh,
                system_prompt=system_prompt, stop_on_fail=stop_on_fail,
                model_name=model_name,
            )
            all_turns.extend(turns)
        return _aggregate(label, len(sessions), all_turns)

    try:
        if base_url is not None:
            return _run_against(base_url)
        with running_server(
            model_path, ctx=ctx, ngl=ngl,
            log_path=server_log_path,
            startup_timeout=server_startup_timeout,
            chat_template_kwargs=chat_template_kwargs,
        ) as url:
            return _run_against(url)
    finally:
        if log_fh is not None:
            log_fh.close()
