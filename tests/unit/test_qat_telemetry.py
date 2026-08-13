"""Unit tests for QAT training telemetry: flip decomposition and log parsing.

The metrics here are the *instrument* for a ternary run — a ternary model can lower its
loss purely by scale drift with no code flips at all, so a wrong flip number is worse
than none. These pin the decomposition (sign reorganization vs densification) and the
stdout parser that turns an in-flight run's log into a time series.
"""

from __future__ import annotations

import torch

from quant_tuner.qat.train import flip_report
from scripts.parse_qat_log import add_flip_velocity, parse, summarize


class FakeLinear:
    def __init__(self, w):
        self.weight = w


class FakeTernary:
    def __init__(self, w):
        self.linear = FakeLinear(w)


class FakeModel:
    def __init__(self, mods):
        self._mods = mods

    def named_modules(self):
        return self._mods.items()


def _snap_and_report(w0, w1, prev=None):
    """Snapshot codes from w0, then report against a model holding w1."""
    from quant_tuner.qat.ternary import ternarize_group

    codes0, scale0, _ = ternarize_group(w0)
    snaps = {"m": (codes0.to(torch.int8).cpu(), scale0.to(torch.float16).cpu())}
    model = FakeModel({"m": FakeTernary(w1)})
    return flip_report(model, snaps, prev=prev)


def test_unchanged_weights_report_zero_flips():
    w = torch.randn(64, 128)
    stats, _ = _snap_and_report(w, w.clone())
    assert stats["m"]["flip_pct"] == 0.0
    assert stats["m"]["zero_to_nonzero"] == 0
    assert stats["m"]["nonzero_to_zero"] == 0
    assert stats["m"]["sign_flip"] == 0
    assert stats["m"]["density"] == stats["m"]["density_start"]


def test_sign_flip_is_counted_separately_from_density_change():
    """Negating the weights flips every nonzero sign at CONSTANT density.

    This is the case that a bare flip-percentage cannot distinguish from mass
    recruitment of dead weights — the whole reason both directions are counted.
    """
    w = torch.randn(64, 128)
    stats, _ = _snap_and_report(w, -w)
    st = stats["m"]
    assert st["sign_flip"] > 0
    assert st["zero_to_nonzero"] == 0
    assert st["nonzero_to_zero"] == 0
    assert st["density"] == st["density_start"]


def test_densification_shows_up_as_zero_to_nonzero():
    """Scaling weights up past the ternary threshold recruits zeros; density must rise."""
    # Explicitly sparse: 10% of entries live. Turning a further 10% on keeps every
    # original weight above the (mean-|w|-derived) threshold, so this is pure recruitment.
    w0 = torch.zeros(64, 128)
    w0.view(-1)[: 64 * 128 // 10] = 1.0
    w1 = w0.clone()
    w1.view(-1)[: 64 * 128 // 5] = 1.0
    stats, _ = _snap_and_report(w0, w1)
    st = stats["m"]
    assert st["zero_to_nonzero"] > 0
    assert st["density"] > st["density_start"]
    assert st["densify_ratio"] is None or st["densify_ratio"] > 1


def test_velocity_delta_needs_a_previous_report():
    w0 = torch.randn(32, 128)
    first, _ = _snap_and_report(w0, w0.clone())
    assert "flip_pct_delta" not in first["m"]
    second, _ = _snap_and_report(w0, -w0, prev=first)
    assert second["m"]["flip_pct_delta"] == second["m"]["flip_pct"] - first["m"]["flip_pct"]


def test_signed_scale_drift_keeps_direction():
    """Mean |ds|/s hides whether scales grow or shrink; the signed number must not."""
    w = torch.randn(64, 128)
    stats, _ = _snap_and_report(w, w * 2.0)
    assert stats["m"]["scale_drift_signed"] > 0
    shrunk, _ = _snap_and_report(w, w * 0.5)
    assert shrunk["m"]["scale_drift_signed"] < 0


LOG = """\
[qat] step 25/522 loss=1.0636 lr=4.62e-04 mem=30.8GiB 356.0s/step
[qat] code flips vs run start:
  model.layers.0.self_attn.q_proj: flips 0.0157% (0->±:927 ±->0:1412) scale-drift 0.60%
  model.layers.35.mlp.down_proj: flips 0.0106% (0->±:3616 ±->0:1708) scale-drift 0.88%
[qat] checkpoint @ step 25: 252 tensors
[qat] step 30/522 loss=5.4884 lr=5.00e-04 mem=30.8GiB 355.6s/step
[qat] step 40 VAL masked-CE 8.6739
[qat] code flips vs run start:
  model.layers.0.self_attn.q_proj: flips 0.2555% (0->±:17102 ±->0:25273) scale-drift 1.13%
  model.layers.35.mlp.down_proj: flips 0.0764% (0->±:23477 ±->0:14984) scale-drift 1.14%
[qat] checkpoint @ step 50: 252 tensors
"""


def test_parse_extracts_steps_val_and_flips():
    d = parse(LOG)
    assert [r["step"] for r in d["steps"]] == [25, 30]
    assert d["steps"][0]["loss"] == 1.0636
    assert d["val"] == [{"step": 40, "val_masked_ce": 8.6739}]
    # flip rows attach to the checkpoint line that FOLLOWS them, not the preceding step
    assert {r["step"] for r in d["flips"]} == {25, 50}


def test_parse_attributes_a_trailing_flip_block_to_the_last_step():
    """A run still in flight prints the flip block before its checkpoint line lands."""
    partial = LOG[: LOG.rindex("[qat] checkpoint @ step 50")]
    d = parse(partial)
    assert [r["step"] for r in d["flips"] if r["step"] != 25] == [30, 30]


def test_flip_velocity_is_the_per_checkpoint_delta():
    d = parse(LOG)
    add_flip_velocity(d["flips"])
    q = [r for r in d["flips"] if r["tensor"].endswith("q_proj")]
    assert q[0]["flip_pct_delta"] is None  # nothing to difference against
    assert q[1]["flip_pct_delta"] == round(0.2555 - 0.0157, 6)
    assert q[1]["z2nz_delta"] == 17102 - 927


def test_densify_ratio_separates_the_two_mechanisms():
    d = parse(LOG)
    rows = {r["tensor"]: r for r in d["flips"] if r["step"] == 50}
    q = rows["model.layers.0.self_attn.q_proj"]
    assert q["densify_ratio"] < 1  # net pruning / sign reorganization
    assert q["net_density_delta"] < 0


def test_summarize_reports_the_loss_peak_and_best_val():
    s = summarize(parse(LOG))
    assert s["loss_peak"] == 5.4884
    assert s["loss_peak_step"] == 30
    assert s["val_best_step"] == 40
    assert s["total_steps"] == 522
