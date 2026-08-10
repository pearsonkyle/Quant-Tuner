"""Red-team attacks, re-paired with generic refusals.

``datasets/redteam-safety-disclosures`` holds one row per adversarial case: the attack
conversation plus **what the target actually replied** — which, for the 26 ``complied`` rows,
is working harmful content, and for all rows includes the target's raw ``target_reasoning``.

Calibrating (or training) on that as-is would teach the exact behavior the disclosure exists
to document. So this module keeps only the *attack side* — the user turns, multi-turn
escalation preserved — and replaces **every** assistant turn with a generic refusal:

* the attack distribution is what we want represented (jailbreak phrasing, roleplay framing,
  encoded payloads), because low-bit quantization is measurably where refusal behavior
  degrades first;
* the response distribution we want represented is refusal, uniformly, regardless of what the
  target did on the day;
* ``target_reasoning`` and the original assistant content are dropped entirely and never
  reach a corpus.

Refusals are drawn deterministically from a small bank rather than one fixed string: a
22-turn crescendo attack would otherwise repeat one sentence eleven times, which is a
degenerate pattern to put in front of an importance matrix. The pick is keyed by
``case_id`` + turn index, so rebuilds are byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

REDTEAM_DATASET = "pearsonkyle/redteam-safety-disclosures"
REDTEAM_DIR = REPO / "datasets" / "redteam-safety-disclosures" / "data"
# `all` (348 rows: 306 defended + 26 complied + 16 errored), not `flagged` (42): we want the
# whole attack distribution. What the target replied is discarded either way, so the
# defended/complied outcome only matters as provenance.
REDTEAM_SPLIT = "all"

SOURCE_REDTEAM = "redteam-refusals"

# Plain, non-preachy refusals: a short decline, no lecture, offering the legitimate
# neighbouring help where there is one. Deliberately varied in length and shape so the
# corpus carries a refusal *register*, not one memorized sentence.
GENERIC_REFUSALS: tuple[str, ...] = (
    "I can't help with that.",
    "I'm not able to help with this one. If there's a related but legitimate task I can "
    "help with, tell me more about what you're actually trying to accomplish.",
    "Sorry — I won't do that.",
    "I can't assist with this request. Happy to help with something else.",
    "That's not something I'll help with. If you're researching this defensively, I can "
    "talk about mitigations and detection instead.",
    "No — I can't provide that.",
    "I'm going to decline this. Let me know if there's another way I can help.",
    "I won't help with this, including in a hypothetical, fictional, or 'for research' "
    "framing. If you have a different question, I'm glad to help with that.",
)


def pick_refusal(key: str) -> str:
    """Deterministic refusal for a (case, turn) key — same input, same corpus bytes."""
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return GENERIC_REFUSALS[h % len(GENERIC_REFUSALS)]


def _parse_messages(raw) -> list[dict]:
    """Rows store ``messages`` as a JSON/py-literal string or a real list."""
    if isinstance(raw, list):
        return [m for m in raw if isinstance(m, dict)]
    if isinstance(raw, str):
        for loader in (json.loads, _literal_eval):
            try:
                v = loader(raw)
            except Exception:  # noqa: BLE001 - try the next loader
                continue
            if isinstance(v, list):
                return [m for m in v if isinstance(m, dict)]
    return []


def _literal_eval(s: str):
    import ast

    return ast.literal_eval(s)


def refusal_messages(row: dict) -> list[dict]:
    """The attack turns with every assistant turn replaced by a generic refusal.

    Returns ``[]`` when the row has no user turn to attack with (nothing to learn from).
    """
    msgs = _parse_messages(row.get("messages"))
    case_id = str(row.get("case_id") or "")
    out: list[dict] = []
    n_asst = 0
    for m in msgs:
        role = m.get("role")
        if role == "user":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                out.append({"role": "user", "content": content})
        elif role == "assistant":
            # The target's own words (and its reasoning) never make it out of the dataset.
            out.append({"role": "assistant",
                        "content": pick_refusal(f"{case_id}:{n_asst}")})
            n_asst += 1
    # Must start on a user turn and end with the refusal, or it isn't a usable exchange.
    while out and out[0]["role"] != "user":
        out.pop(0)
    if not any(m["role"] == "user" for m in out):
        return []
    if out and out[-1]["role"] != "assistant":
        out.append({"role": "assistant", "content": pick_refusal(f"{case_id}:tail")})
    return out


def load_rows(path: Path | None = None, split: str = REDTEAM_SPLIT) -> list[dict]:
    """Rows of the staged red-team disclosure split (local staging copy only).

    Unlike the other published datasets this one is **not** fetched from the Hub on demand:
    both its splits are `publish=False` (they carry working attacks), so a Hub fallback would
    just 404. Absent staging directory -> no red-team source, not an error.
    """
    p = Path(path) if path else REDTEAM_DIR / f"{split}.jsonl"
    if not p.exists():
        return []
    with p.open() as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def refusal_sessions(rows: Iterable[dict]) -> Iterator[dict]:
    """Packer sessions: attack prompts + generic refusals, tagged by vulnerability."""
    for row in rows:
        msgs = refusal_messages(row)
        if not msgs:
            continue
        vuln = str(row.get("vulnerability") or "unknown").lower().replace(" ", "_")
        yield {
            "id": str(row.get("case_id") or ""),
            "group": str(row.get("case_id") or ""),
            "source": f"redteam:{vuln}",
            "messages": msgs,
            "score": 1.0,
            "metrics": {"tool_calls": 0},
            "meta": {
                "vulnerability": row.get("vulnerability"),
                "vulnerability_type": row.get("vulnerability_type"),
                "attack_method": row.get("attack_method"),
                "attack_class": row.get("attack_class"),
                "original_outcome": row.get("outcome"),
                "target_model": row.get("model"),
            },
        }


__all__ = [
    "GENERIC_REFUSALS",
    "REDTEAM_DATASET",
    "REDTEAM_DIR",
    "REDTEAM_SPLIT",
    "SOURCE_REDTEAM",
    "load_rows",
    "pick_refusal",
    "refusal_messages",
    "refusal_sessions",
]
