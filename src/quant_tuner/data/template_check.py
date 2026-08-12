"""Pre-flight: does this model's chat template render TOOL CALLS the way we assume?

Every calibration corpus this repo builds is chat-templated text — the imatrix and the
AWQ/GPTQ activation stats are collected on exactly the byte stream produced by
``tokenizer.apply_chat_template(messages, tools=...)``. That makes the template a silent
dependency of every number downstream, and it has bitten us twice already:

* Qwen3.5-VL's template raises ``"No user query found"`` on a window with no user turn, so
  :func:`quant_tuner.data.split.session_windows` swallowed the exception and dropped whole
  sessions — the calibration corpus quietly fell from 500k to 37k logtrain tokens.
* A template that drops the ``tools=`` argument, or renders ``tool_calls`` as bare prose,
  produces a corpus with no tool-call *structure* in it at all. The imatrix still builds,
  the quant still ships, and the only symptom is that the 2-bit rows can't tool-call.

Neither failure raises. So we check the template up front, on a fixture that exercises the
full tool-calling shape, and record the report in the corpus audit. Run it against a new
model BEFORE spending a day on corpora:

    uv run python scripts/verify_chat_template.py --model out/exp-060/model_extracted

The known-marker table is descriptive, not prescriptive: an unrecognised marker is a
WARNING (we may just not have seen that family yet), while "the arguments never appear" or
"the tool schema never appears" is a hard FAIL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Markers families we have verified render structured tool calls. Purely informational —
# a template using something else is warned about, not rejected.
KNOWN_TOOL_CALL_MARKERS = (
    "<tool_call>",          # Qwen2.5 / Qwen3 / Qwen3.5 / Qwen3.6 family
    "<|tool_call|>",
    "[TOOL_CALLS]",         # Mistral
    "<|python_tag|>",       # Llama 3.x
    "```json",              # some Gemma / generic templates
    "<function_call>",
    "<|tool▁calls▁begin|>",  # DeepSeek
    "functools[",
)

KNOWN_TOOL_RESPONSE_MARKERS = (
    "<tool_response>",
    "<|tool_response|>",
    "[TOOL_RESULTS]",
    "<|tool▁output▁begin|>",
    "ipython",
    "tool\n",               # `<|im_start|>tool` and friends
)

# The fixture is deliberately shaped like the real agentic sessions we calibrate on: two
# tools in scope, an assistant turn that carries BOTH prose and a call, a tool result, and
# a closing assistant turn. `arguments` is a dict — `coerce_tool_call_arguments` normalizes
# the JSON-string form the logs sometimes carry before it ever reaches a template.
FIXTURE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the repository checkout and return its "
                           "combined output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "the command to run"},
                    "timeout_sec": {"type": "integer", "description": "kill after N seconds"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from disk.",
            "parameters": {
                "type": "object",
                # `max_bytes` is the schema probe: this tool is never CALLED in the
                # fixture, so the name can only reach the render through `tools=`. A
                # parameter of the called tool would also appear inside the call's
                # arguments, and the check would pass on a template that drops schemas.
                "properties": {"path": {"type": "string"},
                               "max_bytes": {"type": "integer"}},
                "required": ["path"],
            },
        },
    },
]

FIXTURE_MESSAGES: list[dict] = [
    {"role": "system", "content": "You are a careful software engineering agent."},
    {"role": "user", "content": "Which tests are failing in this repo?"},
    {
        "role": "assistant",
        "content": "Let me run the test suite first.",
        "tool_calls": [
            {
                "id": "call_qt_0",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": {"command": "pytest -q tests/", "timeout_sec": 600},
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_qt_0",
        "name": "bash",
        "content": "2 failed, 118 passed\nFAILED tests/test_split.py::test_windows",
    },
    {
        "role": "assistant",
        "content": "`tests/test_split.py::test_windows` is the failing test.",
    },
]

# A session that STARTS on an assistant turn, i.e. the shape a window boundary produces in
# the middle of a long agentic trajectory. Strict templates reject it; we need to know that
# they do (the packer skips those starts) rather than discover it as a token shortfall.
FIXTURE_ASSISTANT_FIRST: list[dict] = [
    {"role": "system", "content": "You are a careful software engineering agent."},
    *FIXTURE_MESSAGES[2:],
]


@dataclass
class Check:
    name: str
    ok: bool
    severity: str          # "fail" | "warn"
    detail: str = ""

    @property
    def blocking(self) -> bool:
        return not self.ok and self.severity == "fail"


@dataclass
class TemplateReport:
    model: str
    checks: list[Check] = field(default_factory=list)
    render: str = ""
    tool_call_marker: str | None = None
    tool_response_marker: str | None = None

    @property
    def ok(self) -> bool:
        return not any(c.blocking for c in self.checks)

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.severity == "warn"]

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.blocking]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "ok": self.ok,
            "tool_call_marker": self.tool_call_marker,
            "tool_response_marker": self.tool_response_marker,
            "checks": [
                {"name": c.name, "ok": c.ok, "severity": c.severity, "detail": c.detail}
                for c in self.checks
            ],
        }

    def summary(self) -> str:
        lines = [f"chat-template tool-call report: {self.model}"]
        for c in self.checks:
            mark = "PASS" if c.ok else ("FAIL" if c.severity == "fail" else "WARN")
            lines.append(f"  [{mark}] {c.name}" + (f" — {c.detail}" if c.detail else ""))
        lines.append(f"  => {'OK' if self.ok else 'NOT USABLE'}"
                     f" ({len(self.failures)} failing, {len(self.warnings)} warnings)")
        return "\n".join(lines)


def _first_present(text: str, candidates: tuple[str, ...]) -> str | None:
    return next((c for c in candidates if c in text), None)


def _render(tok, messages: list[dict], tools: list[dict] | None) -> str:
    return tok.apply_chat_template(messages, tools=tools, tokenize=False)


def check_template(tok, model_label: str = "") -> TemplateReport:
    """Run every tool-call rendering check against ``tok``'s chat template."""
    from quant_tuner.data.ingest import coerce_tool_call_arguments
    from quant_tuner.data.split import session_windows, stub_tools

    label = model_label or getattr(tok, "name_or_path", "") or "<tokenizer>"
    rep = TemplateReport(model=label)
    add = rep.checks.append

    if not getattr(tok, "chat_template", None):
        add(Check("has_chat_template", False, "fail",
                  "tokenizer exposes no chat_template — every corpus would be raw text"))
        return rep
    add(Check("has_chat_template", True, "fail"))

    messages = json.loads(json.dumps(FIXTURE_MESSAGES))
    coerce_tool_call_arguments(messages)

    try:
        text = _render(tok, messages, FIXTURE_TOOLS)
    except Exception as e:  # noqa: BLE001
        add(Check("renders_tool_session", False, "fail", f"{type(e).__name__}: {e}"))
        return rep
    rep.render = text
    add(Check("renders_tool_session", True, "fail"))

    # --- the tool SCHEMAS must reach the corpus -----------------------------------
    # Without them the model is calibrated on calls it was never shown the signature for.
    missing_names = [t["function"]["name"] for t in FIXTURE_TOOLS
                     if t["function"]["name"] not in text]
    add(Check("tool_schemas_rendered", not missing_names, "fail",
              f"tool names absent from render: {missing_names}" if missing_names
              else "both tool names present"))
    add(Check("tool_parameters_rendered", "max_bytes" in text, "fail",
              "a declared parameter of a tool that is never called ('max_bytes') is "
              "absent — the template is dropping the `tools=` parameter schemas"
              if "max_bytes" not in text else "schema parameters reach the corpus"))

    # --- the CALL must be structured, not prose -----------------------------------
    rep.tool_call_marker = _first_present(text, KNOWN_TOOL_CALL_MARKERS)
    add(Check("tool_call_marker", rep.tool_call_marker is not None, "warn",
              f"marker: {rep.tool_call_marker!r}" if rep.tool_call_marker
              else f"no known marker found; expected one of {KNOWN_TOOL_CALL_MARKERS}"))
    add(Check("tool_call_arguments_rendered", "pytest -q tests/" in text, "fail",
              "the tool_call arguments never appear in the render — the template is "
              "dropping assistant.tool_calls" if "pytest -q tests/" not in text else ""))
    name_rendered = '"bash"' in text or "'bash'" in text or "\nbash" in text
    add(Check("tool_call_name_rendered", name_rendered, "warn",
              "" if name_rendered
              else "the called tool's name is not obviously in the render"))

    # --- the RESULT must be attributable to the call ------------------------------
    result_rendered = "2 failed, 118 passed" in text
    add(Check("tool_result_rendered", result_rendered, "fail",
              "" if result_rendered
              else "role=tool content never appears — tool results are being dropped"))
    rep.tool_response_marker = _first_present(text, KNOWN_TOOL_RESPONSE_MARKERS)
    add(Check("tool_response_marker", rep.tool_response_marker is not None, "warn",
              f"marker: {rep.tool_response_marker!r}" if rep.tool_response_marker
              else "tool results are not wrapped in any known role marker"))

    # --- the render must tokenize back to SPECIAL ids -----------------------------
    # transformers encodes in-text special tokens as single ids by default; if that stops
    # holding, calibration sees `<`, `|`, `im_start` ... as ordinary BPE and the whole
    # corpus is off-distribution relative to inference.
    # Cover both the tokenizer's declared specials and the marker families — Qwen keeps
    # <tool_call> out of all_special_tokens, and it is exactly the marker we care about.
    declared = list(tok.all_special_tokens or []) + list(
        getattr(tok, "additional_special_tokens", []) or [])
    specials = sorted({s for s in declared + list(KNOWN_TOOL_CALL_MARKERS)
                       + list(KNOWN_TOOL_RESPONSE_MARKERS)
                       if s and s.startswith("<") and s in text})
    bad = [s for s in specials
           if len(tok(s, add_special_tokens=False)["input_ids"]) != 1]
    add(Check("special_tokens_single_id", not bad, "fail",
              f"these markers tokenize as multiple ids: {bad}" if bad
              else f"{len(specials)} in-text special tokens each map to one id"))

    # --- stubbed schemas must still render ----------------------------------------
    # stratified_pack renders stub_tools() in every window past the quota. A template that
    # rejects an empty `properties` object would drop ~every window but the first.
    try:
        stub_text = _render(tok, messages, stub_tools(FIXTURE_TOOLS))
        ok_stub = "bash" in stub_text
        detail = "" if ok_stub else "stubbed schema rendered without the tool name"
    except Exception as e:  # noqa: BLE001
        ok_stub, detail = False, f"{type(e).__name__}: {e} (schema dedup would drop windows)"
    add(Check("stub_tools_render", ok_stub, "fail", detail))

    # --- reasoning: where does this template keep it? ------------------------------
    # Not pass/fail — the answer decides how the corpus must be cut. Measured on Qwen3.6:
    # kept on the final assistant turn, scrubbed from history, and an EMPTY <think></think>
    # is emitted on the final turn when no reasoning is supplied.
    rsn_msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "QT_RSN_HISTORY"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2", "reasoning_content": "QT_RSN_FINAL"},
    ]
    try:
        rsn_text = _render(tok, rsn_msgs, None)
        final_kept = "QT_RSN_FINAL" in rsn_text
        history_kept = "QT_RSN_HISTORY" in rsn_text
    except Exception as e:  # noqa: BLE001
        rsn_text, final_kept, history_kept = f"<{e}>", False, False
    add(Check("reasoning_final_turn_kept", final_kept, "warn",
              "reasoning survives on the render's final assistant turn — which is what "
              "universal.reasoning_windows relies on to get any reasoning into the corpus"
              if final_kept else
              "this template drops reasoning even on the FINAL assistant turn: no corpus "
              "cut can carry chain-of-thought through it. Calibrate with "
              "reasoning_policy='drop' and treat reasoning coverage as 0."))
    add(Check("reasoning_history_scrubbed", not history_kept, "warn",
              "history reasoning is scrubbed (expected; matches inference)" if not history_kept
              else "this template KEEPS reasoning in history — reasoning_windows is "
                   "unnecessary here, and the ordinary windows already carry it"))

    # Qwen3.6 added a `preserve_thinking` flag that keeps reasoning on EVERY assistant turn
    # instead of only the final one. Where it exists, a corpus can carry far more
    # reasoning-mode text than `reasoning_windows` can recover — at the cost of a history
    # distribution that differs from default inference, which strips it. Detected, not used
    # automatically: that trade-off is a judgement call per release.
    try:
        pt_text = tok.apply_chat_template(rsn_msgs, tools=None, tokenize=False,
                                          preserve_thinking=True)
        pt_supported = "QT_RSN_HISTORY" in pt_text
    except Exception:  # noqa: BLE001 - templates that don't take the kwarg
        pt_supported = False
    add(Check("preserve_thinking_supported", pt_supported, "warn",
              "template keeps history reasoning under preserve_thinking=True — far more "
              "reasoning-mode text is available than reasoning_windows can recover, if you "
              "accept a history distribution that differs from default inference"
              if pt_supported else
              "no preserve_thinking support; reasoning_windows is the only way to get "
              "chain-of-thought into the corpus (expected for Qwen3.5 and earlier)"))

    # --- windowing must survive an assistant-first body ---------------------------
    wins = session_windows(tok, json.loads(json.dumps(FIXTURE_MESSAGES)), FIXTURE_TOOLS,
                           cap_tokens=4096, system_content=None, max_windows=4)
    add(Check("session_windows_nonempty", bool(wins), "fail",
              f"{len(wins)} window(s) emitted" if wins
              else "session_windows produced NOTHING for a normal tool session — the "
                   "packer would silently skip every session of this shape"))

    strict_wins = session_windows(tok, json.loads(json.dumps(FIXTURE_ASSISTANT_FIRST)),
                                  FIXTURE_TOOLS, cap_tokens=4096, max_windows=4)
    add(Check("assistant_first_window", bool(strict_wins), "warn",
              "template refuses a window that starts on an assistant turn (strict, like "
              "Qwen3.5-VL). The packer skips those starts; expect fewer windows per "
              "session — verify the corpus hits its token target."
              if not strict_wins else f"{len(strict_wins)} window(s)"))

    return rep


def assert_template_ok(tok, model_label: str = "") -> TemplateReport:
    """:func:`check_template`, raising on any blocking failure."""
    rep = check_template(tok, model_label)
    if not rep.ok:
        raise RuntimeError(
            "chat template cannot carry tool calls into the calibration corpus:\n"
            + rep.summary()
        )
    return rep


__all__ = [
    "FIXTURE_ASSISTANT_FIRST",
    "FIXTURE_MESSAGES",
    "FIXTURE_TOOLS",
    "KNOWN_TOOL_CALL_MARKERS",
    "KNOWN_TOOL_RESPONSE_MARKERS",
    "Check",
    "TemplateReport",
    "assert_template_ok",
    "check_template",
]
