"""The QAT backend abstraction.

These are cheap invariants, but they are the ones whose violation is *silent*. The
trainer's device line was `"mps" if mps.is_available() else "cpu"` for its whole life:
on a CUDA box that selects CPU, raises nothing, and runs ~100x slow. Nothing downstream
notices — the loss falls, the checkpoints save. So the properties worth pinning are the
ones a wrong answer does not announce.
"""

from __future__ import annotations

import pytest
import torch

from quant_tuner.qat._device import MPS_MAX_WINDOW, resolve_backend


@pytest.mark.parametrize("name", ["cuda", "mps", "cpu"])
def test_every_backend_is_constructible_without_that_device_present(name):
    """Resolution must not require the hardware — a CPU-only CI box still has to be able
    to introspect the CUDA spec (this is what the trainer's startup banner reads)."""
    b = resolve_backend(name)
    assert b.name == name
    assert isinstance(b.foreach, bool)
    assert isinstance(b.teacher_dtype, torch.dtype)


def test_mps_keeps_its_two_workarounds_and_no_one_else_inherits_them():
    """The two MPS quirks that a port must not carry over: foreach kernels deadlock at
    full-model scale, and the unfused Metal SDPA caps the window at n_heads*S^2 < 2^31."""
    mps = resolve_backend("mps")
    assert mps.foreach is False
    assert mps.needs_chunked_sdpa is True
    assert mps.max_window == MPS_MAX_WINDOW

    for name in ("cuda", "cpu"):
        b = resolve_backend(name)
        assert b.foreach is True, f"{name} must not inherit the MPS foreach deadlock"
        assert b.needs_chunked_sdpa is False
        assert b.max_window is None, f"{name} has no MPSGraph element cap"


def test_cuda_does_not_release_the_allocator_cache_on_a_cadence():
    """The every-5-steps release exists because macOS OOM-kills a run whose working set
    creeps into swap. On CUDA that failure mode does not exist and the release costs a
    device sync plus blocks the allocator would have reused, so it must default off."""
    assert resolve_backend("mps").default_empty_cache_every == 5
    assert resolve_backend("cuda").default_empty_cache_every == 0


def test_teacher_dtype_is_per_backend():
    assert resolve_backend("mps").teacher_dtype is torch.float16  # no bf16 on M1-gen
    assert resolve_backend("cuda").teacher_dtype is torch.bfloat16
    assert resolve_backend("cpu").teacher_dtype is torch.float32


def test_an_explicit_device_index_survives_resolution():
    """Memory counters are per-device; dropping the index would report device 0's numbers
    for a model that lives on device 1."""
    b = resolve_backend("cuda:1")
    assert b.name == "cuda:1"
    assert b.is_cuda


def test_auto_picks_an_accelerator_when_one_is_present():
    b = resolve_backend("auto")
    if torch.cuda.is_available():
        assert b.name == "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        assert b.name == "mps"
    else:
        assert b.name == "cpu"


def test_memory_probes_are_safe_on_the_live_backend():
    """empty_cache/peak/reset must be callable unconditionally — the trainer calls them
    on every checkpoint and validation regardless of device."""
    b = resolve_backend("auto")
    b.reset_peak()
    b.empty_cache()
    assert b.allocated_gib() >= 0.0
    assert b.peak_gib() >= 0.0
    assert b.total_gib() >= 0.0
    assert b.describe()


def test_an_unknown_device_is_refused_rather_than_silently_defaulted():
    with pytest.raises(ValueError, match="unsupported QAT device"):
        resolve_backend("tpu")


# ------------------------------------------------------- bf16 compute reads fp32 masters
def _tiny_ternary_model():
    """A two-linear stack wrapped exactly the way train_qat wraps the real model."""
    from quant_tuner.qat.ternary import TernaryLinear

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False),
                              torch.nn.Linear(128, 128, bias=False))
    for i in (0, 1):
        net[i] = TernaryLinear(net[i], trainable=True)
    return net


def test_flip_telemetry_reads_the_tensor_that_gets_exported():
    """Under --compute-dtype bf16 the live weight is a bf16 copy of an fp32 master, and
    export_qat ternarizes the masters. If the telemetry read the copy instead, flips would
    be recorded against a model that is never exported — and flip velocity, not loss, is
    how a ternary run is judged."""
    from quant_tuner.qat.master_opt import MasterOptimizer
    from quant_tuner.qat.ternary import ternarize_group
    from quant_tuner.qat.train import latent_weights, snapshot_codes

    net = _tiny_ternary_model()
    params = [m.linear.weight for m in net]
    opt = MasterOptimizer(params, lambda ms: torch.optim.SGD(ms, lr=0.0))
    net.to(torch.bfloat16)

    lat = latent_weights(net, opt)
    assert set(lat) == {"0", "1"}, lat
    for name, master in lat.items():
        assert master.dtype is torch.float32, "telemetry must see fp32, not the bf16 copy"
        assert master is not dict(net.named_modules())[name].linear.weight

    # Move one master by a hair and DO NOT propagate it to the bf16 copy — exactly what a
    # sub-ulp optimizer step does. The two views of "the latent" now disagree.
    with torch.no_grad():
        opt.masters[0] += opt.masters[0].abs().mean() * 0.5
    snap_master = snapshot_codes(net, k=2, latents=lat)
    snap_live = snapshot_codes(net, k=2)

    codes_master, _, _ = ternarize_group(opt.masters[0].float())
    codes_live, _, _ = ternarize_group(net[0].linear.weight.detach().float())
    assert torch.equal(snap_master["0"][0], codes_master.to(torch.int8).cpu()), \
        "telemetry must ternarize the fp32 master — the tensor export_qat reads"
    assert torch.equal(snap_live["0"][0], codes_live.to(torch.int8).cpu())
    assert not torch.equal(snap_master["0"][0], snap_live["0"][0]), \
        "the two views must actually differ, or this test proves nothing"


def test_latent_weights_is_empty_without_master_optimizer():
    """In the fp32 path the live weight already IS the latent — no indirection, and the
    telemetry must not silently gain one."""
    from quant_tuner.qat.train import latent_weights

    net = _tiny_ternary_model()
    opt = torch.optim.SGD([m.linear.weight for m in net], lr=0.0)
    assert latent_weights(net, opt) == {}
