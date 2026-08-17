"""The unattended chain's decisions — the ones that run with nobody watching.

`choose_stop_weight` sets a hyper-parameter for a ~33 h training run, and
`analyze_swe_anomalies` is what tells a mute model apart from an incapable one. Both are
read by a script that fires hours after anyone last looked at it, so their failure modes
have to be pinned rather than eyeballed once.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


csw = _load("choose_stop_weight")
asa = _load("analyze_swe_anomalies")
reg = _load("qat_registry")


# --------------------------------------------------------------- stop-weight choice
def test_mute_model_does_not_raise_the_stop_weight():
    """The failure that motivated the ablation. A model that already stops too early must
    never have its stop signal weighted UP — that is how sft32k(6.0) was produced."""
    probe = {"sentence_period": 0.974, "after_tool_call": 0.953}
    w, why = csw.decide(probe, {"mode": "mute"})
    assert w == csw.DEFAULT_WEIGHT
    assert any("over-weighted" in r for r in why)


def test_looping_model_raises_the_stop_weight():
    w, _ = csw.decide({"sentence_period": 0.009, "after_tool_call": 0.2}, {})
    assert w == csw.RAISED_WEIGHT


def test_trajectory_alone_can_diagnose_a_loop_the_probe_calls_healthy():
    """Vanilla is the case that proves the probe is not sufficient: its single-token
    probe is textbook healthy at both points, and it still loops for 19 turns."""
    healthy_probe = {"sentence_period": 0.0092, "after_tool_call": 0.99995}
    assert csw.decide(healthy_probe, {})[0] == csw.DEFAULT_WEIGHT
    assert csw.decide(healthy_probe, {"mode": "loop"})[0] == csw.RAISED_WEIGHT


def test_contradictory_evidence_holds_the_natural_rate_and_says_so():
    """Stops early at a sentence AND fails to stop after a tool call is not a weight
    problem — raising the weight would deepen the early stopping."""
    w, why = csw.decide({"sentence_period": 0.9, "after_tool_call": 0.1}, {})
    assert w == csw.DEFAULT_WEIGHT
    assert any("CONTRADICTORY" in r for r in why)


def test_no_measurement_defaults_but_labels_itself_a_fallback():
    w, why = csw.decide({}, {})
    assert w == csw.DEFAULT_WEIGHT
    assert any("NOT a measurement" in r for r in why)


def test_healthy_model_is_left_alone():
    w, why = csw.decide({"sentence_period": 0.01, "after_tool_call": 0.99},
                        {"mode": "worked, unresolved"})
    assert w == csw.DEFAULT_WEIGHT
    assert any("healthy band" in r for r in why)


# ------------------------------------------------------------------ anomaly modes
def _write(tmp: Path, label: str, result: dict, traj: list[dict]) -> Path:
    import json
    (tmp / f"result_{label}.json").write_text(json.dumps(result))
    (tmp / f"traj_{label}.json").write_text(json.dumps(traj))
    return tmp


def test_mute_is_distinguished_from_incapacity(tmp_path):
    """1 output token and 0 tool calls is a termination failure. Collapsing it into
    `resolved=0` alongside a genuine attempt is what makes a pass-rate column useless
    for steering training."""
    _write(tmp_path, "M", {"out_tokens": 1, "resolved": 0, "patch_produced": 0}, [])
    r = asa.analyze("M", tmp_path)
    assert r["mode"] == "mute"
    assert any("TERMINATION failure" in f for f in r["flags"])


def test_repeated_command_is_a_loop(tmp_path):
    traj = [{"cmd": "ls", "kind": "ok"} for _ in range(6)]
    _write(tmp_path, "L", {"out_tokens": 900, "resolved": 0}, traj)
    r = asa.analyze("L", tmp_path)
    assert r["mode"] == "loop"
    assert r["longest_repeat_run"] == 6


def test_alternating_commands_still_count_as_a_loop(tmp_path):
    """The real vanilla trajectory never repeats twice IN A ROW — it alternates two
    commands — so a consecutive-run test alone would miss it."""
    traj = [{"cmd": "cat f" if i % 2 else "cat f#L437", "kind": "ok"} for i in range(19)]
    _write(tmp_path, "A", {"out_tokens": 1781, "resolved": 0}, traj)
    r = asa.analyze("A", tmp_path)
    assert r["mode"] == "loop"
    assert r["longest_repeat_run"] < asa.LOOP_RUN     # not caught by the run test
    assert r["distinct_commands"] == 2


def test_a_real_attempt_is_not_flagged_as_a_failure_mode(tmp_path):
    traj = [{"cmd": f"cmd{i}", "kind": "ok"} for i in range(10)]
    _write(tmp_path, "W", {"out_tokens": 2295, "resolved": 0, "patch_produced": 1}, traj)
    assert asa.analyze("W", tmp_path)["mode"] == "patched, unresolved"


def test_resolved_wins(tmp_path):
    traj = [{"cmd": f"c{i}", "kind": "ok"} for i in range(10)]
    _write(tmp_path, "R", {"out_tokens": 2295, "resolved": 1, "patch_produced": 1}, traj)
    assert asa.analyze("R", tmp_path)["mode"] == "resolved"


@pytest.mark.parametrize("s,expected", [
    ("abababababababababababababab", True),
    ("the quick brown fox jumps over the lazy dog again", False),
    ("ls -la /tmp", False),
    ("echo hi; " * 12, True),
])
def test_degenerate_text_detection(s, expected):
    assert asa.degenerate_text(s) is expected


# ------------------------------------------------------------------- registry legs
def test_legs_recover_the_true_starting_loss(tmp_path):
    """Reading only the surviving log reports a resumed leg's mid-run loss as the run's
    first — off by 350 steps and, for sft32k, by a precision change too."""
    run = tmp_path / "trained_x"
    run.mkdir()
    (run / "train.diverged-run1.log").write_text(
        "[qat] step 1/613 loss=0.6836 lr=0.00e+00 gnorm=1.0 mem=31.6/70.6GiB 60.0s/step\n"
        "[qat] step 280/613 loss=8.4090 lr=1.0e-04 gnorm=9.0 mem=31.6/70.6GiB 60.0s/step\n")
    (run / "train.log").write_text(
        "[qat] resumed at step 350 (mi=350) with adafactor state\n"
        "[qat] step 355/613 loss=0.7676 lr=2.36e-04 gnorm=0.70 mem=31.6/70.6GiB 63.5s/step\n"
        "[qat] done at step 613: loss 0.684 -> 0.905\n")
    legs = reg.read_legs(run)
    assert [x["fate"] for x in legs] == ["diverged", "complete"]
    assert legs[0]["loss_first"] == 0.6836
    assert legs[0]["step_first"] == 1
    assert legs[1]["resumed_from"] == 350
    assert legs[1]["complete"] is True


def test_a_run_with_no_legs_is_not_a_crash(tmp_path):
    run = tmp_path / "trained_empty"
    run.mkdir()
    assert reg.read_legs(run) == []
