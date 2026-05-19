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

try:
    from jsonschema import Draft7Validator
    _HAVE_JSONSCHEMA = True
except ImportError:  # pragma: no cover
    _HAVE_JSONSCHEMA = False


# Argument keys that hold free-form natural language or open-ended code/shell
# strings. Two reasonable agents will rarely emit byte-identical values for
# these, so we score them as presence-only when present.
_FREE_TEXT_ARG_KEYS = frozenset({
    "command", "description", "content", "message", "text", "prompt",
    "question", "questions", "new_string", "old_string", "code", "body",
    "instructions", "explanation", "summary", "thought", "thinking",
    "query", "search",
})

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


def is_schema_valid(
    tool_name: str, args: dict | None, tools: list[dict]
) -> tuple[bool, str]:
    """Validate ``args`` against the tool's declared JSON schema.

    Returns ``(ok, message)``. ``message`` is a short human-readable diagnostic.
    """
    if not any(
        (t.get("function") or {}).get("name") == tool_name or t.get("name") == tool_name
        for t in tools
    ):
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


def _shortrepr(v: Any, maxlen: int = 120) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= maxlen else s[:maxlen] + "..."


def param_score(
    pred_args: dict | None, truth_args: dict, schema: dict | None
) -> tuple[float, dict]:
    """Fraction of required keys that are present and value-equivalent to truth.

    A key counts as a hit if it's present *and* its value compares as either
    exactly equal or semantically similar (see ``compare_value``). Free-text
    keys count as hits when merely present.
    """
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
