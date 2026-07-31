"""Unit tests for the red-team safety dataset builder — pure record shaping.

No model, no server: exercises `iter_redteam_records` against a fake workspace
of disclosure JSONs + per-case CSVs, pinning the three things the dataset must
carry — model id, full conversation, outcome — plus the dual-use default that
every split is withheld from the Hub.
"""

from __future__ import annotations

import csv
import json

from quant_tuner.datasets.redteam_disclosures import _messages, iter_redteam_records
from quant_tuner.datasets.registry import get_spec


def _write_disclosure(ws, model, findings):
    (ws / f"disclosure_{model}_rep1.json").write_text(json.dumps({
        "target_model": model, "judge_model": "judge", "simulator_model": "sim",
        "config": "red_team_broad", "findings": findings,
    }))


def _write_per_case(ws, model, rows):
    path = ws / f"{model}_per_case.csv"
    cols = ["model", "case_id", "category", "vulnerability", "vulnerability_type",
            "attack_method", "attack_class", "score", "input", "actual_output", "reason"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# _messages
# ---------------------------------------------------------------------------


def test_messages_single_turn_is_user_then_assistant():
    assert _messages("attack", None, "sure, here") == [
        {"role": "user", "content": "attack"},
        {"role": "assistant", "content": "sure, here"},
    ]


def test_messages_preserves_multi_turn_history():
    convo = [{"role": "user", "content": "step1"}, {"role": "assistant", "content": "ok"},
             {"role": "user", "content": "step2"}]
    msgs = _messages("step1", convo, "final answer")
    # every escalation turn is kept, then the final response appended
    assert [m["content"] for m in msgs] == ["step1", "ok", "step2", "final answer"]


def test_messages_does_not_duplicate_seed_when_it_is_the_trailing_turn():
    convo = [{"role": "assistant", "content": "a"}, {"role": "user", "content": "seed"}]
    msgs = _messages("seed", convo, "resp")
    assert sum(m["content"] == "seed" for m in msgs) == 1


def test_messages_skips_empty_turns_and_missing_response():
    assert _messages("q", [{"role": "assistant", "content": None}], None) == [
        {"role": "user", "content": "q"}
    ]


# ---------------------------------------------------------------------------
# iter_redteam_records
# ---------------------------------------------------------------------------


def test_flagged_only_yields_findings_with_model_conversation_outcome(tmp_path):
    _write_disclosure(tmp_path, "modelX", [
        {"case_id": "c1", "severity": "complied", "category": "Safety",
         "vulnerability": "Toxicity", "vulnerability_type": "profanity",
         "attack_method": "Prompt Injection", "attack_class": "single-turn",
         "seed_prompt": "do bad", "conversation": [], "target_response": "ok bad",
         "judge_reason": "complied", "target_reasoning": "cot", "error": None},
    ])
    recs = list(iter_redteam_records(flagged_only=True, workspaces=[tmp_path]))
    assert len(recs) == 1
    r = recs[0]
    assert r["model"] == "modelX"                     # model id
    assert r["outcome"] == "complied" and r["unsafe"] is True   # outcome
    assert r["messages"][0]["content"] == "do bad"    # conversation
    assert r["messages"][-1]["content"] == "ok bad"
    assert r["target_reasoning"] == "cot"


def test_errored_finding_is_not_marked_unsafe(tmp_path):
    _write_disclosure(tmp_path, "m", [
        {"case_id": "e1", "severity": "errored", "seed_prompt": "x",
         "conversation": [], "target_response": None, "error": "timeout"},
    ])
    r = list(iter_redteam_records(flagged_only=True, workspaces=[tmp_path]))[0]
    assert r["outcome"] == "errored" and r["unsafe"] is False


def test_all_split_adds_defended_from_per_case(tmp_path):
    _write_disclosure(tmp_path, "m", [
        {"case_id": "c1", "severity": "complied", "seed_prompt": "bad",
         "conversation": [], "target_response": "bad out"},
    ])
    _write_per_case(tmp_path, "m", [
        {"model": "m", "case_id": "c1", "score": "0", "input": "bad",
         "actual_output": "bad out", "category": "Safety"},          # same case (deduped)
        {"model": "m", "case_id": "d1", "score": "1", "input": "safe ask",
         "actual_output": "I can't help", "category": "Safety", "reason": "refused"},
    ])
    flagged = list(iter_redteam_records(flagged_only=True, workspaces=[tmp_path]))
    full = list(iter_redteam_records(flagged_only=False, workspaces=[tmp_path]))
    assert {r["outcome"] for r in flagged} == {"complied"}
    outcomes = sorted(r["outcome"] for r in full)
    assert outcomes == ["complied", "defended"]       # defended added, c1 not doubled


def test_dedup_across_disclosure_and_per_case(tmp_path):
    """A finding present in both sources appears once (disclosure wins — richer)."""
    _write_disclosure(tmp_path, "m", [
        {"case_id": "c1", "severity": "complied", "seed_prompt": "bad",
         "conversation": [], "target_response": "r", "target_reasoning": "why"},
    ])
    _write_per_case(tmp_path, "m", [
        {"model": "m", "case_id": "c1", "score": "0", "input": "bad", "actual_output": "r"},
    ])
    full = list(iter_redteam_records(flagged_only=False, workspaces=[tmp_path]))
    assert len(full) == 1
    assert full[0]["target_reasoning"] == "why"       # kept the disclosure's richer row


def test_missing_workspace_is_empty_not_error(tmp_path):
    assert list(iter_redteam_records(workspaces=[tmp_path / "nope"])) == []


# ---------------------------------------------------------------------------
# registry wiring + dual-use default
# ---------------------------------------------------------------------------


def test_registered_and_both_splits_withheld_by_default():
    spec = get_spec("redteam-safety-disclosures")
    assert {s.name for s in spec.splits} == {"flagged", "all"}
    # dual-use: NOTHING auto-publishes — the harmful completions stay off the Hub
    assert all(s.publish is False for s in spec.splits)
    assert spec.default_split == "flagged"
