"""Record builder for the red-team safety dataset.

Turns the artifacts written by the red-team eval

    <workspace>/disclosure_<model>_repN.json   # findings: attack + response + reasoning + verdict
    <workspace>/results_per_case.csv           # every attempted case (incl. defended/safe)

into flat records that track exactly what the user asked for: **the target model
id, the full conversation history, and the outcome**. Same shape/tooling as
``swe_trajectories.py`` so it plugs into the dataset registry with a one-entry
``DatasetSpec``.

One row = one attack case:

    {
      "model": "ornith-1.0-35b",                 # the TARGET that was probed
      "case_id": "0fde225510a6842e",             # content hash — joins across runs
      "messages": [ {"role": "user", ...},       # the full conversation (multi-turn
                    {"role": "assistant", ...} ], #  jailbreaks carry every turn)
      "outcome": "complied" | "defended" | "errored",
      "unsafe": true|false,                      # complied == the safety hole
      "category" / "vulnerability" / "vulnerability_type" / "attack_method" / "attack_class",
      "judge_reason": "...",                      # why the judge scored it so
      "judge_model" / "simulator_model" / "config",
      "target_reasoning": "...",                  # the model's own CoT (findings only)
    }

The ``complied`` rows are the training-relevant safety holes: pair each attack
with a *refusal* target and mix with benign→helpful and you have the QAT
fine-tune set that actually generalizes (see ``docs/lens.md`` — a single steering
vector does not). The ``defended`` rows are the positive examples (the model
already refuses these; keep it that way).

**Dual-use note.** ``complied`` rows contain a working attack *and* the harmful
completion it elicited. That is exactly what a model's authors need to fix it,
and exactly what should not be broadcast. Every split here defaults to
``publish=False`` for that reason — build locally, share via a *private* Hub repo
(`scripts/dataset.py push --private`) or a metadata-only view. This is authorized
defensive work on the operator's own models.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Default source: the broad sweep run in this repo.
DEFAULT_WORKSPACES = [REPO / "out" / "redteam" / "broad"]

_OUTCOME = {"1": "defended", "0": "complied", "": "errored"}


def _messages(seed_prompt: str | None, conversation: list | None, response: str | None) -> list[dict]:
    """Full conversation as chat messages: history (if multi-turn) + final response.

    Single-turn attacks have no ``conversation``; the seed prompt is the whole
    user turn. Multi-turn jailbreaks carry every escalation step in ``conversation``
    (already ``{role, content}``), and the seed prompt is not re-appended when it
    is already the trailing user turn.
    """
    msgs: list[dict] = []
    for turn in conversation or []:
        if turn.get("content"):
            msgs.append({"role": turn.get("role") or "user", "content": turn["content"]})
    # `conversation` (when present) already carries the full history including the
    # seed as its first user turn; only fall back to the seed for single-turn cases.
    if not msgs and seed_prompt:
        msgs.append({"role": "user", "content": seed_prompt})
    if response is not None and not (
        msgs and msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == response
    ):
        msgs.append({"role": "assistant", "content": response})
    return msgs


def _from_disclosure(path: Path) -> Iterator[dict]:
    """Records from one disclosure JSON (the findings: complied + errored)."""
    doc = json.loads(path.read_text())
    model = doc.get("target_model")
    common = {
        "model": model,
        "judge_model": doc.get("judge_model"),
        "simulator_model": doc.get("simulator_model"),
        "config": doc.get("config"),
    }
    for f in doc.get("findings", []):
        outcome = "errored" if f.get("severity") == "errored" else "complied"
        yield {
            **common,
            "case_id": f.get("case_id"),
            "messages": _messages(f.get("seed_prompt"), f.get("conversation"), f.get("target_response")),
            "outcome": outcome,
            "unsafe": outcome == "complied",
            "category": f.get("category"),
            "vulnerability": f.get("vulnerability"),
            "vulnerability_type": f.get("vulnerability_type"),
            "attack_method": f.get("attack_method"),
            "attack_class": f.get("attack_class"),
            "judge_reason": f.get("judge_reason"),
            "target_reasoning": f.get("target_reasoning"),
            "error": f.get("error"),
        }


def _from_per_case(path: Path) -> Iterator[dict]:
    """Records from a per-case CSV (every case, incl. defended). No CoT/reasoning."""
    model = path.parent.name  # not authoritative; the CSV carries the real model per row
    for r in csv.DictReader(path.open()):
        outcome = _OUTCOME.get(r.get("score") or "", "errored")
        yield {
            "model": r.get("model") or model,
            "case_id": r.get("case_id"),
            "messages": _messages(r.get("input"), None, r.get("actual_output")),
            "outcome": outcome,
            "unsafe": outcome == "complied",
            "category": r.get("category"),
            "vulnerability": r.get("vulnerability"),
            "vulnerability_type": r.get("vulnerability_type"),
            "attack_method": r.get("attack_method"),
            "attack_class": r.get("attack_class"),
            "judge_reason": r.get("reason"),
            "target_reasoning": None,
            "error": None,
        }


def iter_redteam_records(
    *,
    flagged_only: bool = True,
    workspaces: list[Path] | None = None,
) -> Iterator[dict]:
    """Yield red-team records across ``workspaces``, deduplicated by (model, case_id).

    ``flagged_only=True`` yields just the safety findings (complied + errored),
    sourced from the richer disclosure JSONs (with the model's own reasoning).
    ``flagged_only=False`` adds the ``defended`` cases from the per-case CSVs, so
    the full split covers safe *and* unsafe behavior for balanced analysis.
    """
    seen: set[tuple] = set()

    def _emit(rec: dict) -> Iterator[dict]:
        key = (rec.get("model"), rec.get("case_id"))
        if rec.get("case_id") and key in seen:
            return
        seen.add(key)
        yield rec

    for ws in (workspaces or DEFAULT_WORKSPACES):
        ws = Path(ws)
        if not ws.exists():
            continue
        # findings first (rich), so their reasoning wins over the CSV's thinner row
        for disc in sorted(ws.glob("disclosure_*.json")):
            for rec in _from_disclosure(disc):
                yield from _emit(rec)
        if not flagged_only:
            for pc in sorted(ws.glob("*_per_case.csv")):
                for rec in _from_per_case(pc):
                    if rec["outcome"] == "defended":  # findings already emitted above
                        yield from _emit(rec)
