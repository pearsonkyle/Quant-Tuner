"""External SFT corpora -> the ``sft.jsonl`` schema ``qat.corpus`` already reads.

Three public datasets feed the staged curriculum (see ``docs/ternary_qat_curriculum.md``):

* ``HuggingFaceH4/ultrachat_200k`` — 207,865 multi-turn chats, ``role``/``content`` only.
  No system prompt, no tools, no reasoning. Broad conversational grounding.
* ``r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation`` config ``sft_tools`` — 5,337
  agentic tool trajectories, median 11 turns, 100% carrying real tool schemas.
* the same repo's ``sft_science`` — 10,189 single-turn reasoning rows.

Everything here converts INTO the schema ``data.universal`` writes, so the existing
``qat.corpus.build_sft_corpus`` consumes these with no changes: one JSON object per line
carrying ``id``, ``source``, ``split``, ``messages``, ``tools`` and the ``n_*`` counters.

Four things about these sources are not obvious and each one silently corrupts a corpus:

1. **``sft_tools`` and ``sft_agent`` are the same rows.** Byte-identical parquet
   (4,226,831 bytes) and an identical id-set hash. The Hub's ``/size`` API reports every
   config with the same row count, which makes them look distinct; they are not. Taking
   both double-weights the agentic half. :data:`DISTILL_ALIASES` records this.
2. **Tool schemas store ``parameters_json`` as a JSON STRING**, not a nested object. The
   Qwen3 chat template renders tools with ``tool | tojson``, so passing the row through
   unconverted emits the parameter schema as one escaped string blob — it templates and
   tokenizes fine, and teaches the model a format no inference-time caller produces.
   :func:`normalize_tool` is the fix and is unit-tested.
3. **Reasoning is inline ``<think>`` in content**, never the ``reasoning_content`` field,
   despite the field existing on every message. ``data.reasoning`` already normalizes both
   forms, so this is handled — but a build that keys off ``reasoning_content`` alone
   measures zero reasoning on a corpus that is ~40% reasoning.
4. **``sft_science`` is drawn from benchmark test sets** — SciQ, CommonsenseQA, ARC,
   OpenBookQA, QASC. See :data:`SCIENCE_BENCHMARK_SOURCES`; training on it contaminates
   any multiple-choice eval, which is why it is not in the default curriculum.

The upstream rows carry no split, so :func:`assign_split` derives a deterministic one from
the row id — the same train/test/holdout discipline the log corpora use, so a later eval
on these sources is still held out.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

from . import reasoning as _reasoning

#: The distillation repo. Configs resolve as ``data/<config>/<split>-*.parquet``.
DISTILL_REPO = "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation"
ULTRACHAT_REPO = "HuggingFaceH4/ultrachat_200k"

#: ultrachat ships four splits; ``train_sft`` is the supervised one. ``train_gen`` is the
#: generation split used to build preference data and is NOT a bigger training set.
ULTRACHAT_SPLIT = "train_sft"

#: Configs that are byte-identical to another. Keyed alias -> canonical.
DISTILL_ALIASES = {"sft_agent": "sft_tools"}

#: ``sft_science`` sources that are public benchmark sets. Training on these contaminates
#: the corresponding evals — MMLU-Pro overlaps ARC/OpenBookQA/SciQ material directly.
SCIENCE_BENCHMARK_SOURCES = frozenset({
    "SciQ", "CommonsenseQA", "QASC", "ARC-Easy", "ARC-Challenge", "OpenBookQA",
})

#: Deterministic split fractions, matching the log corpora's 80/10/10.
SPLIT_FRACTIONS = (("train", 0.80), ("test", 0.10), ("holdout", 0.10))

#: The distillation repo ships its OWN train/validation/test as separate parquet files, and
#: every row inside the train file is tagged ``split="train"``. So the row's own field
#: carries no information — which file it came from does. Mapped onto our three names:
#: their validation becomes our ``test`` (what the trainer's --split test val corpus reads)
#: and their test becomes our ``holdout`` (kept back for final eval), preserving the rule
#: that nothing the trainer sees is ever graded on.
UPSTREAM_SPLIT_MAP = {"train": "train", "validation": "test", "test": "holdout"}


def assign_split(row_id: str, *, salt: str = "") -> str:
    """Stable train/test/holdout for a row that arrives without one.

    Hashed rather than positional so that re-downloading, re-sharding or filtering the
    upstream file cannot move a row across the boundary — the eval holdout has to mean the
    same thing on every rebuild.
    """
    h = hashlib.sha256(f"{salt}{row_id}".encode()).digest()
    x = int.from_bytes(h[:8], "big") / float(1 << 64)
    acc = 0.0
    for name, frac in SPLIT_FRACTIONS:
        acc += frac
        if x < acc:
            return name
    return SPLIT_FRACTIONS[-1][0]


def normalize_tool(tool: dict) -> dict:
    """One tool schema -> the ``{"type","function":{"name","description","parameters"}}``
    shape the chat template renders.

    The distillation rows nest the parameter schema as ``parameters_json``, a STRING. The
    template's ``tool | tojson`` would emit that string escaped inside the tool block, so
    the model learns a doubly-encoded schema it will never see at inference. Anything
    already in the right shape passes through untouched.
    """
    fn = dict(tool.get("function") or {})
    if "parameters_json" in fn:
        raw = fn.pop("parameters_json")
        if isinstance(raw, str) and raw.strip():
            try:
                fn["parameters"] = json.loads(raw)
            except json.JSONDecodeError:
                # A schema we cannot parse is worse than none: it would render as a string.
                fn["parameters"] = {"type": "object", "properties": {}}
        elif "parameters" not in fn:
            fn["parameters"] = {"type": "object", "properties": {}}
    fn.setdefault("parameters", {"type": "object", "properties": {}})
    fn.setdefault("description", f"Tool function: {fn.get('name', 'unknown')}")
    return {"type": "function", "function": fn}


def _clean_message(msg: dict) -> dict:
    """Drop upstream-only keys and empty fields the template would choke on."""
    out: dict[str, Any] = {"role": msg.get("role"), "content": msg.get("content") or ""}
    for k in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
        v = msg.get(k)
        if v:
            out[k] = v
    return out


def _counters(messages: list[dict], tools: list[dict] | None) -> dict:
    n_calls = sum(len(m.get("tool_calls") or []) for m in messages)
    n_results = sum(1 for m in messages if m.get("role") == "tool")
    return {
        "n_messages": len(messages),
        "n_chars": sum(len(m.get("content") or "") for m in messages),
        "n_tool_calls": n_calls,
        "n_tool_results": n_results,
        "n_reasoning": _reasoning.count_available(messages),
        "tools": tools or None,
    }


def _record(*, rid: str, source: str, messages: list[dict],
            tools: list[dict] | None, meta: dict, split: str) -> dict:
    return {
        "id": rid, "source": source, "split": split,
        "messages": messages, "meta": meta, **_counters(messages, tools),
    }


def convert_distill_rows(rows: list[dict], *, source: str, split: str = "train",
                         drop_benchmarks: bool = False) -> Iterator[dict]:
    """Distillation parquet rows -> sft records.

    ``split`` is the UPSTREAM split the rows were read from (``train``/``validation``/
    ``test``), mapped through :data:`UPSTREAM_SPLIT_MAP`. It is passed in rather than read
    off the row because every row in the train file is tagged ``"train"`` regardless.

    ``drop_benchmarks`` removes rows whose upstream ``source`` is a public benchmark set
    (:data:`SCIENCE_BENCHMARK_SOURCES`); use it whenever the run will later be graded on a
    multiple-choice eval.
    """
    mapped = UPSTREAM_SPLIT_MAP.get(split, split)
    for r in rows:
        upstream = r.get("source") or ""
        if drop_benchmarks and upstream in SCIENCE_BENCHMARK_SOURCES:
            continue
        msgs = r.get("messages")
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        msgs = [_clean_message(m) for m in (msgs or [])]
        if len(msgs) < 2:
            continue
        raw_tools = r.get("tools")
        if isinstance(raw_tools, str):
            raw_tools = json.loads(raw_tools) if raw_tools.strip() else []
        tools = [normalize_tool(t) for t in (raw_tools or [])] or None
        rid = str(r.get("id") or hashlib.sha256(
            json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:24])
        yield _record(
            rid=rid, source=source, messages=msgs, tools=tools,
            meta={"upstream_source": upstream, "domain": r.get("domain"),
                  "sampling_weight": r.get("sampling_weight"),
                  "upstream_split": split},
            split=mapped,
        )


def convert_ultrachat_rows(rows: list[dict], *, source: str = "ultrachat") -> Iterator[dict]:
    """ultrachat_200k rows -> sft records. No tools, no reasoning, no system turn."""
    for r in rows:
        msgs = [_clean_message(m) for m in (r.get("messages") or [])]
        if len(msgs) < 2:
            continue
        rid = str(r.get("prompt_id") or hashlib.sha256(
            json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:24])
        yield _record(
            rid=rid, source=source, messages=msgs, tools=None,
            meta={"upstream_source": ULTRACHAT_REPO},
            split=assign_split(rid, salt=source),
        )
