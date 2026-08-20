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


# --- architecture section -------------------------------------------------------

def test_arch_spec_reads_shapes_and_flags_the_qwen3_delta():
    """Shapes come from the model's own config; the ✻ marks where it leaves stock Qwen3."""
    from scripts.qat_report import arch_spec

    cfg = {"hidden_size": 4096, "intermediate_size": 12288, "num_attention_heads": 32,
           "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 36,
           "vocab_size": 151669, "max_position_embeddings": 65536, "hidden_act": "silu",
           "rms_norm_eps": 1e-6, "rope_theta": 1e6, "tie_word_embeddings": False}
    html = arch_spec(cfg)
    assert "36" in html and "4096 / 12288" in html
    assert "GQA 4:1" in html                      # 32 heads / 8 kv
    assert f"{36 * 7}" in html                    # ternary linears
    assert "✻" in html and "Qwen/Qwen3-8B" in html

    stock = {**cfg, "vocab_size": 151936, "max_position_embeddings": 40960}
    assert "✻" not in arch_spec(stock)            # no delta -> no footnote


def test_arch_card_falls_back_to_live_embed_on_a_placeholder(monkeypatch):
    """hfviewer answers 200 with a ~950-byte 'unavailable' SVG for unindexed repos.

    Inlining that would freeze a broken graph into the report, so it must fall back to
    the live <img>, which starts working by itself once the graph exists.
    """
    from scripts import qat_report

    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body.encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    placeholder = '<svg aria-label="Graph temporarily unavailable"></svg>'
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp(placeholder))
    out = qat_report.arch_card("prism-ml/Nope")
    assert "<img" in out and "hfviewer.com/api/card.svg" in out
    assert "aria-label=\"Graph temporarily" not in out   # not inlined

    real = '<svg viewBox="0 0 480 300">' + "<rect/>" * 400 + "</svg>"
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp(real))
    out = qat_report.arch_card("prism-ml/Yes")
    assert out.count("<svg") == 1 and "<img" not in out  # inlined, self-contained


def test_arch_card_survives_no_network(monkeypatch):
    from scripts import qat_report

    def boom(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    out = qat_report.arch_card("prism-ml/Whatever")
    assert "<img" in out  # degrades to the live embed rather than raising


def test_parse_reads_the_kd_step_line():
    """KD runs insert `kl=` between loss and lr; the pre-KD regex silently parsed ZERO
    steps from such a log (bit the kd8b-full report on 2026-08-18)."""
    rows = parse(
        "[qat] step 10/613 loss=0.8744 kl=0.5535 lr=1.67e-04 gnorm=1.42 "
        "mem=31.6/70.6GiB 45.9s/step\n"
        "[qat] step 15/613 loss=0.8306 lr=4.53e-04 gnorm=1.00 mem=31.6GiB 46.2s/step\n"
    )["steps"]
    assert [r["step"] for r in rows] == [10, 15]
    assert rows[0]["kd_kl"] == 0.5535 and rows[0]["loss"] == 0.8744
    assert rows[1]["kd_kl"] is None          # non-KD line still parses, kl empty


def test_parse_reads_the_stop_anchor_step_line():
    rows = parse(
        "[qat] step 10/613 loss=0.8744 kl=0.5535 an=0.0121 lr=1.67e-04 gnorm=1.42 "
        "mem=31.6/70.6GiB 45.9s/step\n")["steps"]
    assert rows[0]["kd_kl"] == 0.5535 and rows[0]["stop_anchor"] == 0.0121


def test_parse_reads_the_steer_step_line():
    """Third occurrence of the same failure class: every new step-line field between
    loss= and lr= silently zeroes n_steps_logged. an= bit kd8b-full; st= bit anchor6."""
    rows = parse(
        "[qat] step 10/613 loss=0.87 kl=0.55 an=0.01 st=0.12 lr=1.67e-04 gnorm=1.4 "
        "mem=31.6/70.6GiB 45.9s/step\n")["steps"]
    assert rows and rows[0]["steer"] == 0.12 and rows[0]["kd_kl"] == 0.55


def test_parse_reads_the_rep_steer_step_line():
    rows = parse(
        "[qat] step 10/613 loss=0.87 kl=0.55 an=0.01 st=0.12 rp=0.03 lr=1.67e-04 "
        "gnorm=1.4 mem=31.6/70.6GiB 45.9s/step\n")["steps"]
    assert rows and rows[0]["steer_rep"] == 0.03 and rows[0]["steer"] == 0.12


def test_step_line_with_rep_kl():
    """rk= (rep teacher-KL, anchor9) must parse — every new step-line field has broken
    STEP_RE silently before (kl=, st=, rp=); same commit, same test, every time."""
    rows = parse(
        "[qat] step 7/613 loss=0.9312 kl=0.5012 an=0.0100 st=0.0002 rp=0.0031 "
        "rk=0.4210 lr=4.99e-04 gnorm=1.10 mem=31.7/88.8GiB 49.3s/step\n")["steps"]
    assert rows and rows[0]["kd_kl"] == 0.5012
    assert rows[0]["steer_rep"] == 0.0031
    assert rows[0]["rep_kl"] == 0.4210
