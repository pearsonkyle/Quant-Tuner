"""Unit tests for imatrix calibrator helpers (no model load required)."""

from pathlib import Path

import numpy as np
import pytest

from quant_tuner.calibrate import imatrix as imx
from quant_tuner.calibrate.imatrix import (
    ForwardStats,
    _col_l2_sq,
    _combine_moe,
    _l1_normalize,
    build_analytic,
    build_hybrid_custom,
    build_outlier_l4,
)
from quant_tuner.models import llama_cpp
from quant_tuner.models.hf_gguf_map import is_moe_expert, is_ssm, map_hf_to_gguf


def test_col_l2_sq_sums_over_rows():
    # GGUF stores weights as [n_out, n_in]. We want ||W[:, c]||^2 per input channel c.
    w = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, -3.0]], dtype=np.float32)  # 2x3
    out = _col_l2_sq(w)
    # cols: 1^2+4^2=17, 2^2+0^2=4, 3^2+3^2=18
    np.testing.assert_allclose(out, [17.0, 4.0, 18.0])


def test_l1_normalize_preserves_zero_vector():
    v = np.zeros(8, dtype=np.float32)
    np.testing.assert_array_equal(_l1_normalize(v), v)


def test_l1_normalize_makes_mean_magnitude_one():
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    n = _l1_normalize(v)
    # Mean magnitude should now be 1.0
    assert abs(float(np.abs(n).mean()) - 1.0) < 1e-6


def test_map_hf_to_gguf_attention():
    assert map_hf_to_gguf("model.layers.3.self_attn.q_proj") == "blk.3.attn_q.weight"
    assert map_hf_to_gguf("model.layers.10.self_attn.o_proj") == "blk.10.attn_output.weight"
    assert map_hf_to_gguf("model.layers.0.self_attn.out_proj") == "blk.0.attn_output.weight"


def test_map_hf_to_gguf_mlp_and_head():
    assert map_hf_to_gguf("model.layers.7.mlp.gate_proj") == "blk.7.ffn_gate.weight"
    assert map_hf_to_gguf("model.layers.7.mlp.up_proj") == "blk.7.ffn_up.weight"
    assert map_hf_to_gguf("model.layers.7.mlp.down_proj") == "blk.7.ffn_down.weight"
    assert map_hf_to_gguf("lm_head") == "output.weight"


def test_map_hf_to_gguf_unmapped_returns_none():
    assert map_hf_to_gguf("model.embed_tokens") is None
    assert map_hf_to_gguf("model.layers.0.self_attn.rotary_emb.inv_freq") is None


def test_is_ssm():
    assert is_ssm("blk.3.ssm_dt.weight")
    assert not is_ssm("blk.3.attn_q.weight")
    assert not is_ssm("blk.3.ffn_down.weight")


def test_map_hf_to_gguf_moe_and_router():
    # Gemma-4 style fused 3D experts + router (a plain Linear).
    assert map_hf_to_gguf("model.layers.5.experts.gate_up_proj") == "blk.5.ffn_gate_up_exps.weight"
    assert map_hf_to_gguf("model.layers.5.experts.down_proj") == "blk.5.ffn_down_exps.weight"
    assert map_hf_to_gguf("model.layers.5.mlp.experts.gate_up_proj") == "blk.5.ffn_gate_up_exps.weight"
    assert map_hf_to_gguf("model.layers.5.router.proj") == "blk.5.ffn_gate_inp.weight"


def test_map_hf_to_gguf_diffusiongemma_trunk_prefixes():
    # DiffusionGemma: backbone under model.decoder.*; encoder stack is tied.
    assert map_hf_to_gguf("model.decoder.layers.2.self_attn.q_proj") == "blk.2.attn_q.weight"
    assert map_hf_to_gguf("model.decoder.layers.2.experts.gate_up_proj") == "blk.2.ffn_gate_up_exps.weight"
    assert (
        map_hf_to_gguf("model.encoder.language_model.layers.2.mlp.up_proj")
        == "blk.2.ffn_up.weight"
    )
    assert map_hf_to_gguf("model.decoder.self_conditioning.gate_proj") == "self_cond_gate.weight"


def test_is_moe_expert():
    assert is_moe_expert("blk.0.ffn_gate_up_exps.weight")
    assert is_moe_expert("blk.0.ffn_down_exps.weight")
    assert is_moe_expert("blk.0.ffn_gate_exps.weight")
    assert not is_moe_expert("blk.0.ffn_gate.weight")
    assert not is_moe_expert("blk.0.ffn_gate_inp.weight")


def test_forward_stats_roundtrip(tmp_path):
    stats = ForwardStats(
        e_a4={
            "blk.0.attn_q.weight": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "blk.0.ffn_down.weight": np.array([1.0, 2.0], dtype=np.float32),
        },
        max_abs={
            "blk.0.attn_q.weight": np.array([1.5, 2.5, 0.5], dtype=np.float32),
            "blk.0.ffn_down.weight": np.array([4.0, 5.0], dtype=np.float32),
        },
    )
    out = tmp_path / "stats.npz"
    stats.save(out)
    loaded = ForwardStats.load(out)
    assert set(loaded.e_a4) == set(stats.e_a4)
    for k in stats.e_a4:
        np.testing.assert_allclose(loaded.e_a4[k], stats.e_a4[k])
        np.testing.assert_allclose(loaded.max_abs[k], stats.max_abs[k])


def test_build_analytic_respects_ssm_passthrough(monkeypatch):
    # Two linear-projection tensors and one SSM tensor.
    base = {
        "blk.0.attn_q.weight": np.array([1.0, 4.0, 9.0], dtype=np.float32),    # E[a^2]
        "blk.0.ffn_down.weight": np.array([2.0, 2.0], dtype=np.float32),
        "blk.0.ssm_dt.weight": np.array([7.0, 7.0, 7.0], dtype=np.float32),
    }
    # W[:, c] columns chosen so ||W[:, c]||^2 is easy to verify.
    weights = {
        "blk.0.attn_q.weight": np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32
        ),  # col sums-of-squares: [1, 1, 2]
        "blk.0.ffn_down.weight": np.array([[3.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        # col sums-of-squares: [9 + 16, 0] = [25, 0]
        "blk.0.ssm_dt.weight": np.zeros((2, 3), dtype=np.float32),
    }
    monkeypatch.setattr(imx, "_load_base_imatrix", lambda _p: base)
    monkeypatch.setattr(imx, "_load_weights", lambda _p: weights)

    out = build_analytic("dummy_f16", "dummy_base")

    # attn_q: [1*1, 1*4, 2*9] = [1, 4, 18]
    np.testing.assert_allclose(out["blk.0.attn_q.weight"], [1.0, 4.0, 18.0])
    # ffn_down: [25*2, 0*2] = [50, 0]
    np.testing.assert_allclose(out["blk.0.ffn_down.weight"], [50.0, 0.0])
    # SSM tensor MUST pass through raw E[a^2], not be reranked.
    np.testing.assert_allclose(out["blk.0.ssm_dt.weight"], [7.0, 7.0, 7.0])


def test_build_outlier_l4_falls_back_when_stats_missing(monkeypatch):
    """If forward stats lack a tensor, that tensor falls back to E[a^2]."""
    base = {
        "blk.0.attn_q.weight": np.array([0.0, 16.0, 81.0], dtype=np.float32),
        "blk.0.attn_k.weight": np.array([0.5, 0.5, 0.5], dtype=np.float32),  # no fwd stats
        "blk.0.ssm_dt.weight": np.array([7.0, 7.0], dtype=np.float32),
    }
    monkeypatch.setattr(imx, "_load_base_imatrix", lambda _p: base)

    stats = ForwardStats(
        e_a4={"blk.0.attn_q.weight": np.array([1.0, 16.0, 81.0], dtype=np.float32)},
        max_abs={"blk.0.attn_q.weight": np.array([2.0, 4.0, 9.0], dtype=np.float32)},
    )
    out = build_outlier_l4("dummy_base", stats)

    np.testing.assert_allclose(out["blk.0.attn_q.weight"], [1.0, 4.0, 9.0])
    np.testing.assert_allclose(out["blk.0.attn_k.weight"], [0.5, 0.5, 0.5])  # E[a^2] fallback
    np.testing.assert_allclose(out["blk.0.ssm_dt.weight"], [7.0, 7.0])  # SSM passthrough


def test_build_analytic_moe_per_expert(monkeypatch):
    """3D expert tensors are reranked per expert and flattened expert-major."""
    tname = "blk.0.ffn_gate_up_exps.weight"
    # 2 experts, n_out=2, n_in=3
    w = np.array(
        [
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],   # expert 0 col ssq: [1, 1, 2]
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],   # expert 1 col ssq: [4, 9, 0]
        ],
        dtype=np.float32,
    )
    ea2 = np.array([[1.0, 4.0, 9.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    monkeypatch.setattr(imx, "_load_base_imatrix", lambda _p: {tname: ea2})
    monkeypatch.setattr(imx, "_load_weights", lambda _p: {tname: w})

    out = build_analytic("dummy_f16", "dummy_base")

    # expert 0: [1*1, 1*4, 2*9]; expert 1: [4*2, 9*2, 0*2], expert-major flat
    np.testing.assert_allclose(out[tname], [1.0, 4.0, 18.0, 8.0, 18.0, 0.0])


def test_build_hybrid_custom_moe_normalizes_per_expert(monkeypatch):
    """L1 normalization must not bleed across experts with different scales."""
    tname = "blk.0.ffn_down_exps.weight"
    # Identical structure, expert 1's activations scaled 1000x: per-expert
    # normalization makes both experts' outputs identical.
    w = np.stack([np.eye(3, dtype=np.float32)] * 2)
    ea2 = np.array([[1.0, 2.0, 3.0], [1000.0, 2000.0, 3000.0]], dtype=np.float32)
    monkeypatch.setattr(imx, "_load_base_imatrix", lambda _p: {tname: ea2})
    monkeypatch.setattr(imx, "_load_weights", lambda _p: {tname: w})

    out = build_hybrid_custom("dummy_f16", "dummy_base")
    flat = out[tname]
    np.testing.assert_allclose(flat[:3], flat[3:], rtol=1e-6)


def test_combine_moe_rejects_misaligned_shapes():
    w = np.zeros((2, 4, 3), dtype=np.float32)
    ea2 = np.zeros((3, 3), dtype=np.float32)  # wrong expert count
    with pytest.raises(ValueError, match="does not line up"):
        _combine_moe(w, ea2, lambda a, b: a * b, "blk.0.ffn_gate_up_exps.weight")


def test_build_outlier_l4_moe_passthrough(monkeypatch):
    """Expert tensors pass through per-expert E[a^2] in outlier variants."""
    tname = "blk.0.ffn_gate_up_exps.weight"
    ea2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    monkeypatch.setattr(imx, "_load_base_imatrix", lambda _p: {tname: ea2})

    out = build_outlier_l4("dummy_base", ForwardStats(e_a4={}, max_abs={}))
    np.testing.assert_allclose(out[tname], [1.0, 2.0, 3.0, 4.0])


def _write_moe_imatrix_gguf(path):
    """Write a minimal llama-imatrix-layout GGUF: one dense entry (scalar count)
    and one stacked-expert entry (per-expert counts, one expert unobserved)."""
    imx._ensure_gguf_py()
    gguf = pytest.importorskip("gguf")

    w = gguf.GGUFWriter(str(path), "imatrix")
    w.add_uint32("imatrix.chunk_count", 2)
    w.add_uint32("imatrix.chunk_size", 512)
    w.add_array("imatrix.datasets", ["test"])
    # dense: 1-D in_sum2, scalar count
    w.add_tensor("blk.0.attn_q.weight.in_sum2", np.array([2.0, 4.0, 8.0], dtype=np.float32))
    w.add_tensor("blk.0.attn_q.weight.counts", np.array([2.0], dtype=np.float32))
    # MoE: [n_expert=3, n_in=2] sums, per-expert counts with expert 2 unseen
    sums = np.array([[2.0, 4.0], [30.0, 60.0], [0.0, 0.0]], dtype=np.float32)
    w.add_tensor("blk.0.ffn_gate_up_exps.weight.in_sum2", sums)
    w.add_tensor(
        "blk.0.ffn_gate_up_exps.weight.counts",
        np.array([[2.0], [10.0], [0.0]], dtype=np.float32),
    )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_load_base_imatrix_moe_per_expert_counts(tmp_path):
    path = tmp_path / "imatrix.gguf"
    _write_moe_imatrix_gguf(path)

    base = imx._load_base_imatrix(path)

    np.testing.assert_allclose(base["blk.0.attn_q.weight"], [1.0, 2.0, 4.0])
    moe = base["blk.0.ffn_gate_up_exps.weight"]
    assert moe.shape == (3, 2)
    np.testing.assert_allclose(moe[0], [1.0, 2.0])   # / count 2
    np.testing.assert_allclose(moe[1], [3.0, 6.0])   # / count 10
    np.testing.assert_allclose(moe[2], [1.0, 1.0])   # unseen -> neutral ones


def test_check_moe_coverage(tmp_path):
    path = tmp_path / "imatrix.gguf"
    _write_moe_imatrix_gguf(path)

    cov = imx.check_moe_coverage(path)
    # dense entries (single count) are not reported
    assert list(cov) == ["blk.0.ffn_gate_up_exps.weight"]
    np.testing.assert_allclose(cov["blk.0.ffn_gate_up_exps.weight"], 2 / 3)


def _capture_imatrix_cmd(monkeypatch, **kwargs) -> list[str]:
    """Run llama_cpp.imatrix with run()/llama_bin stubbed; return the built cmd as strings."""
    captured: dict[str, list] = {}
    monkeypatch.setattr(llama_cpp, "llama_bin", lambda name: Path(f"/bin/{name}"))
    monkeypatch.setattr(llama_cpp, "run", lambda cmd, log=None: captured.setdefault("cmd", cmd))
    llama_cpp.imatrix(Path("m.gguf"), Path("corpus.txt"), Path("o.gguf"), **kwargs)
    return [str(c) for c in captured["cmd"]]


def test_imatrix_passes_parse_special_by_default(monkeypatch):
    # Our corpus is chat-templated; control markers must tokenize as special IDs.
    cmd = _capture_imatrix_cmd(monkeypatch)
    assert "--parse-special" in cmd


def test_imatrix_omits_parse_special_when_disabled(monkeypatch):
    cmd = _capture_imatrix_cmd(monkeypatch, parse_special=False)
    assert "--parse-special" not in cmd
