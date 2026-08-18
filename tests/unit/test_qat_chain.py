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


# ------------------------------------------------------------ in-training stop probe
def test_stop_probe_prompts_are_shared_with_the_gguf_probe():
    """Two copies of the probe text would silently stop being comparable the first time
    one was edited, and comparability across the torch and GGUF paths is the point."""
    from quant_tuner.qat import stop_probe as sp
    gguf = _load("probe_stop_prob")
    assert gguf.PROBE_POINTS == sp.PROBE_POINTS
    assert gguf.SENTENCE == sp.SENTENCE
    assert gguf.STOP_PIECE == sp.STOP_PIECE


def test_stop_probe_names_the_diagnostic_and_the_control():
    from quant_tuner.qat import stop_probe as sp
    names = [n for n, _ in sp.PROBE_POINTS]
    assert sp.DIAGNOSTIC in names and sp.CONTROL in names
    assert sp.DIAGNOSTIC != sp.CONTROL


def test_stop_probe_measure_restores_training_mode():
    """Called mid-training, leaving the model in eval() would silently disable dropout
    for the rest of the run."""
    import torch

    from quant_tuner.qat import stop_probe as sp

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 7)

        def forward(self, ids):
            class Out:
                pass
            o = Out()
            o.logits = torch.zeros(1, ids.shape[1], 7)
            return o

    m = Dummy()
    m.train()
    probe = sp.StopProbe(stop_id=3, prompts=[("start", torch.zeros(1, 5, dtype=torch.long))])
    probe.measure(m, "cpu")
    assert m.training, "measure() must restore train mode"


def test_stopprobe_log_line_parses_without_the_gloss_overwriting_values():
    """The printed line repeats two probes inside a bracketed gloss with reference
    values; parsing must take the real ones."""
    parse = _load("parse_qat_log")
    line = ("[qat] step 25 STOPPROBE start=0.0000 mid_sentence=0.0000 "
            "sentence_period=0.0017 sentence_newline=0.0000 after_tool_call=0.9999  "
            "[diagnostic sentence_period=0.9999 vs vanilla 0.0092; "
            "control after_tool_call=0.0001 vs vanilla 0.99995]")
    m = parse.STOPPROBE_RE.match(line)
    assert m
    body = m.group("body").split("[")[0]
    row = {kv.group("k"): float(kv.group("v")) for kv in parse.KV_RE.finditer(body)}
    assert row["sentence_period"] == 0.0017      # not the gloss's 0.9999
    assert row["after_tool_call"] == 0.9999      # not the gloss's 0.0001


def test_latent_lr_mults_track_group_scale():
    """Flip distance is proportional to the group scale, so the multiplier must be too:
    a tensor with 2x the magnitudes gets ~2x the lr (relative to the median tensor),
    non-2D params stay at 1.0, and the clamp bounds runaway ratios."""
    import torch

    from quant_tuner.qat.train import latent_lr_mults

    g = torch.Generator().manual_seed(0)
    base = torch.randn(4, 256, generator=g) * 0.02
    named = [
        ("small.weight", torch.nn.Parameter(base.clone())),
        ("mid.weight", torch.nn.Parameter(base.clone() * 1.5)),
        ("large.weight", torch.nn.Parameter(base.clone() * 3.0)),
        ("norm.weight", torch.nn.Parameter(torch.ones(64))),          # 1D -> 1.0
        ("odd.weight", torch.nn.Parameter(torch.randn(4, 100))),      # not /128 -> 1.0
    ]
    m = latent_lr_mults(named)
    assert m["norm.weight"] == 1.0 and m["odd.weight"] == 1.0
    # median tensor is the reference
    assert m["mid.weight"] == pytest.approx(1.0, abs=1e-6)
    # small is 1/1.5 of the median; large is 3/1.5 = 2.0 (right at the clamp)
    assert m["small.weight"] == pytest.approx(1 / 1.5, rel=1e-3)
    assert m["large.weight"] == pytest.approx(2.0, rel=1e-3)
    # a 10x tensor clamps rather than running away
    named.append(("huge.weight", torch.nn.Parameter(base.clone() * 15.0)))
    assert latent_lr_mults(named)["huge.weight"] <= 2.0


def test_probe_abort_patience():
    """Hysteresis on both abort guards: a reading back inside the band resets the
    counter (anchor3's abort fired at the trough of an oscillation that had already
    recovered once), while N consecutive violations still abort."""
    from quant_tuner.qat.train import probe_abort_check

    def check(probs, strikes):
        return probe_abort_check(probs, "diag", "ctrl", abort_hi=0.09,
                                 abort_ctrl_lo=0.95, patience=2, strikes=strikes)

    s = {"diag": 0, "ctrl": 0}
    # single trough -> warn-only; recovery resets
    assert check({"diag": 0.0, "ctrl": 0.93}, s) is None and s["ctrl"] == 1
    assert check({"diag": 0.0, "ctrl": 0.99}, s) is None and s["ctrl"] == 0
    # two consecutive -> control abort
    assert check({"diag": 0.0, "ctrl": 0.94}, s) is None
    assert check({"diag": 0.0, "ctrl": 0.90}, s) == "control"
    # diagnostic side, monotone collapse aborts one probe late
    s = {"diag": 0, "ctrl": 0}
    assert check({"diag": 0.10, "ctrl": 1.0}, s) is None
    assert check({"diag": 0.15, "ctrl": 1.0}, s) == "diagnostic"
    # disabled guards never fire, patience 1 = old single-reading behavior
    s = {"diag": 0, "ctrl": 0}
    assert probe_abort_check({"diag": 0.5, "ctrl": 0.5}, "diag", "ctrl",
                             abort_hi=0.0, abort_ctrl_lo=0.0,
                             patience=2, strikes=s) is None
    s = {"diag": 0, "ctrl": 0}
    assert probe_abort_check({"diag": 0.0, "ctrl": 0.90}, "diag", "ctrl",
                             abort_hi=0.09, abort_ctrl_lo=0.95,
                             patience=1, strikes=s) == "control"
    # None probs (probe failure) neither fires nor mutates
    s = {"diag": 1, "ctrl": 1}
    assert probe_abort_check(None, "diag", "ctrl", abort_hi=0.09,
                             abort_ctrl_lo=0.95, patience=2, strikes=s) is None
    assert s == {"diag": 1, "ctrl": 1}
