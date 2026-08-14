"""Unit tests for the gradient-spike guard and the staged MasterOptimizer step.

The guard exists because clipping does NOT prevent a divergence: it rescales the update
direction but still takes a full-size step along it, and it hides the excursion from every
logged number. On sft8k-full that cost ~9 GPU-hours of recovery.
"""

from __future__ import annotations

import torch

from quant_tuner.qat.master_opt import MasterOptimizer
from quant_tuner.qat.train import GradSpikeGuard, lr_at


def _warm(g: GradSpikeGuard, value: float = 1.0, n: int = 25) -> None:
    for _ in range(n):
        assert not g.check(value)


def test_no_skip_before_enough_history():
    """Warmup norms are legitimately large; with no stable median there is nothing to
    compare against, so the guard must stay out of the way."""
    g = GradSpikeGuard(factor=2.0, min_history=20)
    for _ in range(19):
        assert not g.check(1000.0)
    assert g.n_skipped == 0


def test_spike_is_skipped_and_normal_steps_are_not():
    g = GradSpikeGuard(factor=4.0)
    _warm(g)
    assert not g.check(2.0)       # 2x median, under the threshold
    assert g.check(50.0)          # 50x median
    assert g.n_skipped == 1
    assert not g.check(1.1)       # back to normal, guard gets out of the way


def test_skipped_norms_do_not_enter_the_history():
    """Otherwise a sustained excursion drags the median up until the guard stops firing —
    exactly the case it exists to catch."""
    g = GradSpikeGuard(factor=4.0)
    _warm(g)
    for _ in range(30):
        assert g.check(100.0)
    assert g.n_skipped == 30
    assert g.last_median == 1.0   # unmoved by the spikes


def test_non_finite_norm_is_always_skipped():
    g = GradSpikeGuard(factor=4.0)
    assert g.check(float("nan"))
    assert g.check(float("inf"))
    assert g.n_skipped == 2


def test_factor_zero_disables():
    g = GradSpikeGuard(factor=0.0)
    _warm(g)
    assert not g.check(1e6)
    assert g.n_skipped == 0


def test_staged_norm_matches_clip_grad_norm():
    """stage_grads_and_norm must return the same pre-clip norm torch would compute."""
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(8, 128)) for _ in range(3)]
    for p in ps:
        p.grad = torch.randn_like(p)
    # reference: torch's own norm over an independent copy, with a clip threshold so high
    # it cannot clip (clip_grad_norm_ returns the PRE-clip norm either way)
    ref = []
    for p in ps:
        q = torch.nn.Parameter(torch.zeros_like(p))
        q.grad = p.grad.clone()
        ref.append(q)
    expect = float(torch.nn.utils.clip_grad_norm_(ref, 1e9, foreach=False))
    opt = MasterOptimizer(ps, lambda ms: torch.optim.SGD(ms, lr=0.0))
    assert abs(opt.stage_grads_and_norm() - expect) < 1e-3


def test_staged_step_updates_params_like_clip_and_step():
    torch.manual_seed(0)
    a = [torch.nn.Parameter(torch.randn(4, 128)) for _ in range(2)]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    grads = [torch.randn_like(p) for p in a]
    for p, g in zip(a, grads, strict=True):
        p.grad = g.clone()
    for p, g in zip(b, grads, strict=True):
        p.grad = g.clone()

    oa = MasterOptimizer(a, lambda ms: torch.optim.SGD(ms, lr=0.1))
    ob = MasterOptimizer(b, lambda ms: torch.optim.SGD(ms, lr=0.1))
    oa.clip_and_step(1.0)
    ob.stage_grads_and_norm()
    ob.step_staged(1.0)
    for x, y in zip(a, b, strict=True):
        assert torch.allclose(x, y, atol=1e-6)


def test_skipping_leaves_params_untouched():
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(4, 128))]
    before = ps[0].detach().clone()
    ps[0].grad = torch.randn_like(ps[0]) * 1e6
    opt = MasterOptimizer(ps, lambda ms: torch.optim.SGD(ms, lr=0.1))
    g = GradSpikeGuard(factor=4.0)
    _warm(g)
    norm = opt.stage_grads_and_norm()
    assert g.check(norm)           # the spike is caught
    opt.zero_grad()                # and the caller skips step_staged
    assert torch.equal(ps[0], before)


def test_warmup_frac_is_honored():
    """The divergence began 4 steps after warmup ended, so this knob has to work."""
    total = 1000
    assert lr_at(50, total, 1.0, warmup_frac=0.05) == 1.0      # end of a 5% ramp
    assert lr_at(50, total, 1.0, warmup_frac=0.20) < 0.3       # still ramping at 20%
    assert lr_at(200, total, 1.0, warmup_frac=0.20) == 1.0
    # monotone non-decreasing through the ramp
    ramp = [lr_at(s, total, 1.0, 0.2) for s in range(0, 200, 10)]
    assert all(b >= a for a, b in zip(ramp, ramp[1:], strict=False))
