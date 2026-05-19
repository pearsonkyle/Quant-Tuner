"""End-to-end tool-call evaluation: per-turn replay + full-session rollouts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from openai import OpenAI

from quant_tuner.eval.scoring import (
    is_schema_valid,
    param_score,
    parse_arguments,
    schema_for,
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
    """Drop non-OpenAI fields and reshape to chat/completions message format."""
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
            out.append(o)
        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id") or "call_0",
                "content": m.get("content", "") or "",
            })
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
) -> Any:
    """Call ``/v1/chat/completions`` with OpenAI + llama.cpp sampling params."""
    return client.chat.completions.create(
        model="local",
        messages=messages,
        tools=tools,
        **sampling.to_request_kwargs(),
    )


# ---------------------------------------------------------------------------
# Eval loops
# ---------------------------------------------------------------------------


def eval_per_turn(
    client: OpenAI,
    session: dict,
    sampling: Sampling,
    *,
    max_turns: int,
    log_fh: IO[str] | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    stop_on_fail: bool = True,
) -> list[dict]:
    """Score every assistant ``tool_calls`` turn in ``session``.

    For each ground-truth assistant turn, replay the prior context and compare
    the model's first tool call to the recorded one. Emits one record per
    scored turn, optionally appended as JSONL to ``log_fh``.
    """
    results: list[dict] = []
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
        truth_args = parse_arguments(
            (truth_tc.get("function") or {}).get("arguments") or truth_tc.get("arguments")
        ) or {}

        try:
            resp = call_model(client, prefix, tools, sampling)
            choice = resp.choices[0].message
            pred_tcs = choice.tool_calls or []
        except Exception as e:
            rec = {
                "session": sid, "source": src, "turn": turn_idx, "error": str(e),
                "truth_name": truth_name, "selection": False,
                "param_acc": 0.0, "schema_valid": False,
            }
            results.append(rec)
            if log_fh is not None:
                log_fh.write(json.dumps(rec) + "\n")
            if stop_on_fail:
                break
            continue

        if not pred_tcs:
            rec = {
                "session": sid, "source": src, "turn": turn_idx,
                "truth_name": truth_name, "pred_name": None,
                "selection": False, "param_acc": 0.0, "schema_valid": False,
                "note": "no tool_calls emitted",
                "pred_content": (choice.content or "")[:200],
            }
            results.append(rec)
            if log_fh is not None:
                log_fh.write(json.dumps(rec) + "\n")
            if stop_on_fail:
                break
            continue

        pred_tc = pred_tcs[0]
        pred_name = pred_tc.function.name
        pred_args = parse_arguments(pred_tc.function.arguments)

        selection = (pred_name or "").lower() == (truth_name or "").lower()
        sch = schema_for(truth_name, tools)
        pacc, pdetails = (
            param_score(pred_args, truth_args, sch) if selection
            else (0.0, {"wrong_tool": True})
        )
        schema_ok, schema_msg = is_schema_valid(pred_name, pred_args, tools)

        rec = {
            "session": sid, "source": src, "turn": turn_idx,
            "truth_name": truth_name, "pred_name": pred_name,
            "selection": selection, "param_acc": pacc, "param_details": pdetails,
            "schema_valid": schema_ok, "schema_msg": schema_msg,
            "truth_args_keys": sorted(truth_args.keys()),
            "pred_args_keys": sorted((pred_args or {}).keys()),
        }
        results.append(rec)
        if log_fh is not None:
            log_fh.write(json.dumps(rec) + "\n")
        if stop_on_fail and not selection:
            break
    return results


def eval_rollout(
    client: OpenAI,
    session: dict,
    sampling: Sampling,
    *,
    max_turns: int,
    log_fh: IO[str] | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> dict:
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

    for _ in range(max_turns):
        try:
            resp = call_model(client, convo, tools, sampling)
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
    """Aggregate metrics returned by :func:`run_toolcall_eval`."""

    model: str
    n_sessions: int
    n_turns: int
    tool_selection_acc: float
    param_acc_mean: float
    schema_valid_rate: float
    rollout_complete_rate: float
    rollout_tool_set_match_rate: float
    per_turn: list[dict] = field(default_factory=list)
    rollouts: list[dict] = field(default_factory=list)
    by_source: dict[str, dict] = field(default_factory=dict)


def _load_sessions(holdout: Path) -> list[dict]:
    with holdout.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _aggregate(
    model_label: str,
    n_sessions: int,
    per_turn: list[dict],
    rollouts: list[dict],
) -> EvalSummary:
    n = len(per_turn)
    n_sel = sum(1 for t in per_turn if t.get("selection"))
    n_schema = sum(1 for t in per_turn if t.get("schema_valid"))
    n_roll_done = sum(1 for r in rollouts if r.get("completed"))
    n_roll_match = sum(1 for r in rollouts if r.get("tool_set_match"))
    pacc = (sum(t.get("param_acc", 0.0) for t in per_turn) / n) if n else 0.0

    by_src: dict[str, list[dict]] = {}
    for t in per_turn:
        by_src.setdefault(t.get("source") or "unknown", []).append(t)
    src_summary = {
        src: {
            "n": len(ts),
            "tool_selection_acc": sum(1 for t in ts if t.get("selection")) / len(ts),
            "param_acc_mean": sum(t.get("param_acc", 0.0) for t in ts) / len(ts),
            "schema_valid_rate": sum(1 for t in ts if t.get("schema_valid")) / len(ts),
        }
        for src, ts in by_src.items()
        if ts
    }

    return EvalSummary(
        model=model_label,
        n_sessions=n_sessions,
        n_turns=n,
        tool_selection_acc=(n_sel / n) if n else 0.0,
        param_acc_mean=pacc,
        schema_valid_rate=(n_schema / n) if n else 0.0,
        rollout_complete_rate=(n_roll_done / len(rollouts)) if rollouts else 0.0,
        rollout_tool_set_match_rate=(n_roll_match / len(rollouts)) if rollouts else 0.0,
        per_turn=per_turn,
        rollouts=rollouts,
        by_source=src_summary,
    )


def run_toolcall_eval(
    holdout: Path,
    *,
    model_path: Path | None = None,
    base_url: str | None = None,
    sampling: Sampling | None = None,
    model_label: str | None = None,
    max_turns_per_session: int = 8,
    rollout_max_turns: int = 12,
    skip_rollout: bool = False,
    stop_on_fail: bool = True,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    ctx: int = 8192,
    ngl: int = 99,
    server_log_path: Path | None = None,
    server_startup_timeout: float = 120.0,
    per_turn_log: Path | None = None,
    progress: bool = False,
) -> EvalSummary:
    """Run the per-turn + rollout passes against one model and return aggregates.

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
        client = OpenAI(base_url=url, api_key="sk-no-key")
        all_turns: list[dict] = []
        all_rolls: list[dict] = []
        for i, sess in enumerate(sessions, 1):
            if progress:
                print(f"[{i}/{len(sessions)}] {sess.get('session_id')}", flush=True)
            turns = eval_per_turn(
                client, sess, sampling,
                max_turns=max_turns_per_session, log_fh=log_fh,
                system_prompt=system_prompt, stop_on_fail=stop_on_fail,
            )
            all_turns.extend(turns)
            if not skip_rollout:
                roll = eval_rollout(
                    client, sess, sampling,
                    max_turns=rollout_max_turns, log_fh=log_fh,
                    system_prompt=system_prompt,
                )
                all_rolls.append(roll)
        return _aggregate(label, len(sessions), all_turns, all_rolls)

    try:
        if base_url is not None:
            return _run_against(base_url)
        with running_server(
            model_path, ctx=ctx, ngl=ngl,
            log_path=server_log_path,
            startup_timeout=server_startup_timeout,
        ) as url:
            return _run_against(url)
    finally:
        if log_fh is not None:
            log_fh.close()
