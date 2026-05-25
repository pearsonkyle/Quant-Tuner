"""Type-aware comparison of predicted vs. ground-truth tool-call arguments.

These functions decide whether a model's tool call counts as correct. Two
calls can be byte-different and still semantically equivalent (a normalized
path, a re-ordered shell flag, a paraphrased natural-language argument), so
``compare_value`` dispatches by argument-key heuristic and JSON-schema type.

See ``docs/benchmarks.md`` § "Tool-call accuracy" for the metric definitions.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from typing import Any

# Argument keys that hold filesystem paths. Compared with path normalization,
# case-insensitive fallback, and basename overlap.
_PATH_ARG_KEYS = frozenset({
    "file_path", "filepath", "filename", "path", "dir", "directory",
    "target_file", "source", "destination",
})

# Argument keys that hold shell commands. Compared with shlex parsing.
_COMMAND_ARG_KEYS = frozenset({"command", "cmd", "shell_command"})


def parse_arguments(args: Any) -> dict | None:
    """Coerce a tool-call ``arguments`` field to a dict, or ``None`` if not parseable."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            v = json.loads(args)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


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
    union = len(sa | sb)
    if not union:
        return False, False
    return False, (len(sa & sb) / union) >= 0.3


def _compare_generic_string(a: Any, b: Any) -> tuple[bool, bool]:
    if a == b and not (isinstance(a, str) and a == ""):
        return True, True
    if not (isinstance(a, str) and isinstance(b, str)):
        return False, False
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return False, False
    return False, (len(ta & tb) / len(ta | tb)) >= 0.7


def compare_value(
    key: str, pred: Any, truth: Any, key_schema: dict | None
) -> tuple[bool, bool, str]:
    """Compare a predicted argument value to the ground truth.

    Returns ``(exact, similar, method)``. ``method`` names the comparator used
    so logs can explain why a value was credited or dinged.
    """
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


def schema_for(tool_name: str, tools: list[dict]) -> dict | None:
    """Look up a tool's parameter schema in an OpenAI-style ``tools`` list."""
    for t in tools:
        fn = t.get("function") or {}
        if fn.get("name") == tool_name or t.get("name") == tool_name:
            return fn.get("parameters") or t.get("parameters")
    return None


_SCHEMA_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _check_value_against_schema(key: str, value: Any, prop: dict) -> str | None:
    """Return a one-line failure reason, or None if value satisfies ``prop``."""
    declared = prop.get("type")
    if isinstance(declared, str):
        allowed = _SCHEMA_TYPE_MAP.get(declared)
        if allowed is not None:
            # bool is an int subclass — reject bool when integer/number expected.
            if declared in ("integer", "number") and isinstance(value, bool):
                return f"{key}: expected {declared}, got boolean"
            if not isinstance(value, allowed):
                return f"{key}: expected {declared}, got {type(value).__name__}"
    enum = prop.get("enum")
    if enum is not None and value not in enum:
        return f"{key}: value not in enum {enum}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        mn, mx = prop.get("minimum"), prop.get("maximum")
        if mn is not None and value < mn:
            return f"{key}: {value} < minimum {mn}"
        if mx is not None and value > mx:
            return f"{key}: {value} > maximum {mx}"
    return None


def is_schema_valid(
    tool_name: str, args: dict | None, tools: list[dict]
) -> tuple[bool, str]:
    """Validate ``args`` against the tool's declared parameter schema.

    Checks: tool is declared, args parsed as object, all ``required`` keys
    present, and for each present key (whether required or not) the value's
    type, ``enum`` membership, and numeric ``minimum``/``maximum`` constraints.
    A tool with no schema (or no ``properties``) passes the per-key checks
    automatically.
    """
    if not any(
        (t.get("function") or {}).get("name") == tool_name or t.get("name") == tool_name
        for t in tools
    ):
        return False, f"tool '{tool_name}' not in tools list"
    if args is None:
        return False, "arguments did not parse as JSON object"
    sch = schema_for(tool_name, tools) or {}
    missing = [k for k in (sch.get("required") or []) if k not in args]
    if missing:
        return False, f"missing required: {missing}"
    props = (sch.get("properties") or {}) if isinstance(sch, dict) else {}
    reasons: list[str] = []
    for k, v in args.items():
        prop = props.get(k)
        if not isinstance(prop, dict):
            continue
        reason = _check_value_against_schema(k, v, prop)
        if reason:
            reasons.append(reason)
    if reasons:
        return False, "; ".join(reasons)
    return True, "ok"


def _shortrepr(v: Any, maxlen: int = 120) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= maxlen else s[:maxlen] + "..."


def param_score(
    pred_args: dict | None, truth_args: dict, schema: dict | None
) -> tuple[float, dict]:
    """Fraction of ground-truth keys present in ``pred_args`` with matching values.

    The denominator is always ``len(truth_args)``. Extra keys in ``pred_args``
    are ignored (no penalty). A truth key counts as a hit if it appears in
    ``pred_args`` AND its value is either exactly equal or semantically
    similar (see ``compare_value``). Free-text values are scored via the same
    ``compare_value`` dispatch — no presence-only credit.
    """
    if pred_args is None:
        return 0.0, {"pred_unparseable": True}
    required = list(truth_args.keys())
    if not required:
        return 1.0, {"required": []}
    props = (schema or {}).get("properties") or {}
    hits = 0
    details: dict[str, Any] = {}
    for k in required:
        if k not in pred_args:
            details[k] = "missing"
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


def score_turn(
    pred_calls: list[tuple[str, dict | None]],
    truth_calls: list[tuple[str, dict]],
    tools: list[dict],
) -> dict:
    """Score one assistant turn that may contain parallel tool calls.

    Each call is a ``(name, args)`` tuple. Returns:

    * ``selection``: ``set(pred_names) == set(truth_names)`` (case-insensitive).
      Strict — extra or missing tools both fail.
    * ``param_acc``: when ``selection`` is True, mean of per-truth-call
      ``param_score`` after greedy first-occurrence name-matching against
      ``pred_calls``. When False, 0.0.
    * ``schema_valid``: ``all(is_schema_valid(name, args, tools))`` across
      every emitted pred call. Empty pred → False.
    * ``param_details``: per-truth-call breakdown for the log.
    """
    truth_names_l = [(n or "").lower() for n, _ in truth_calls]
    pred_names_l = [(n or "").lower() for n, _ in pred_calls]
    selection = bool(truth_names_l) and set(truth_names_l) == set(pred_names_l)

    pacc = 0.0
    param_details: list[dict] = []
    if selection:
        consumed = [False] * len(pred_calls)
        per_call_scores: list[float] = []
        for tname, targs in truth_calls:
            tname_l = (tname or "").lower()
            match_idx = next(
                (i for i, pname_l in enumerate(pred_names_l)
                 if not consumed[i] and pname_l == tname_l),
                None,
            )
            if match_idx is None:
                per_call_scores.append(0.0)
                param_details.append({"truth_name": tname, "result": "no pred match"})
                continue
            consumed[match_idx] = True
            sch = schema_for(tname, tools)
            score, det = param_score(pred_calls[match_idx][1], targs, sch)
            per_call_scores.append(score)
            param_details.append({
                "truth_name": tname,
                "pred_idx": match_idx,
                "score": score,
                "detail": det,
            })
        pacc = sum(per_call_scores) / len(per_call_scores) if per_call_scores else 0.0
    else:
        param_details.append({
            "result": "selection mismatch",
            "truth_names": [n for n, _ in truth_calls],
            "pred_names": [n for n, _ in pred_calls],
        })

    if not pred_calls:
        schema_ok = False
        schema_msg = "no tool_calls emitted"
    else:
        ok_msgs = [is_schema_valid(pn, pa, tools) for pn, pa in pred_calls]
        schema_ok = all(ok for ok, _ in ok_msgs)
        schema_msg = "; ".join(m for _, m in ok_msgs)

    return {
        "selection": selection,
        "param_acc": pacc,
        "param_details": param_details,
        "schema_valid": schema_ok,
        "schema_msg": schema_msg,
        "n_truth_calls": len(truth_calls),
        "n_pred_calls": len(pred_calls),
    }
