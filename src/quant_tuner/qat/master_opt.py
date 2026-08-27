"""fp32-master optimizer wrapper for bf16-compute QAT (the "fp32-master trick").

Raw bf16 latents cannot be trained: a per-step update of ~lr (1e-4-ish) is below
one bf16 ulp of a ~1e-2 latent, so the add is lost and no code ever crosses the
TWN threshold ("no codes flip"); at LRs large enough to register, bf16 training
destabilizes. The correct scheme keeps the *trainable latents* in fp32 master
copies owned by the optimizer while the model's live parameters (and therefore
the forward/backward, activations, and gradients) run in bf16:

    backward -> bf16 grads -> upcast into master.grad (fp32, freeing each bf16
    grad as it is copied) -> global-norm clip in fp32 -> inner per-tensor step
    on the masters -> write masters back into the bf16 params.

Sub-ulp updates accumulate in the fp32 masters across steps and cross code-flip
thresholds exactly as full-fp32 training does; only the *forward* sees bf16
rounding (≤0.4% relative on the TWN scales — codes are unaffected at step 0 and
export reads the fp32 masters, so the artifact stays exact).

Note this is NOT what transformers' Adafactor does internally for bf16 params —
that is a transient upcast-per-step with a bf16 writeback, which still loses
sub-ulp updates between steps. Persistent masters are required.

Device safety: the grad staging and writeback are plain per-tensor python loops on
every backend. The multi-tensor (``foreach``) path is opt-in via the ``foreach``
argument, because MPS's foreach kernels deadlock at full-model scale while CUDA's are
a real speedup over ~250 individual small-kernel launches. See :mod:`quant_tuner.qat._device`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

InnerFactory = Callable[[list[torch.nn.Parameter]], torch.optim.Optimizer]


class MasterOptimizer:
    """Wraps a per-tensor optimizer so it steps fp32 masters of ``params``.

    ``make_inner`` receives the fp32 master parameters and must construct the
    inner optimizer over them (e.g. ``lambda ms: Adafactor(ms, lr=...)``).
    Masters are cloned from the params at construction time — build this BEFORE
    casting the model to bf16 so the masters capture full-precision values.
    """

    def __init__(self, params: Sequence[torch.nn.Parameter], make_inner: InnerFactory,
                 foreach: bool = False):
        self.params = list(params)
        self.foreach = foreach
        self.masters = [
            torch.nn.Parameter(p.detach().to(torch.float32).clone(), requires_grad=False)
            for p in self.params
        ]
        self.inner = make_inner(self.masters)

    @property
    def param_groups(self):  # lr scheduling pokes pg["lr"] — forward to the inner opt
        return self.inner.param_groups

    def clip_and_step(self, max_norm: float | None = 1.0) -> float:
        """Upcast grads into the masters, clip globally in fp32, step, write back.

        Returns the PRE-clip global grad norm. That number is the diagnostic for a
        divergence: clipping hides the excursion from the loss curve, so a run that
        blows up shows only "loss went up" unless the norm is recorded. Returns 0.0
        when no grads are present.
        """
        for p, m in zip(self.params, self.masters, strict=True):
            if p.grad is None:
                m.grad = None
            else:
                m.grad = p.grad.detach().to(torch.float32)
                p.grad = None  # free the bf16 grad immediately (peak-memory trim)
        norm = 0.0
        if max_norm is not None:
            norm = float(torch.nn.utils.clip_grad_norm_(self.masters, max_norm,
                                                        foreach=self.foreach))
        self.inner.step()
        with torch.no_grad():
            for p, m in zip(self.params, self.masters, strict=True):
                p.copy_(m.to(p.dtype))
                m.grad = None
        return norm

    def stage_grads_and_norm(self) -> float:
        """Upcast grads into the masters and return the pre-clip global norm, NO step.

        Split out of `clip_and_step` so a caller can decide whether to step at all — a
        spike guard needs the norm before committing, and re-deriving it afterwards would
        mean either a second pass over every gradient or a step already taken.
        """
        for p, m in zip(self.params, self.masters, strict=True):
            if p.grad is None:
                m.grad = None
            else:
                m.grad = p.grad.detach().to(torch.float32)
                p.grad = None
        # Accumulate the squared norm ON DEVICE and synchronize once. A `.cpu()` per
        # tensor is a device->host sync per tensor: at all-36 that is ~250 stalls per
        # optimizer step, which on CUDA costs more than the arithmetic it is measuring.
        grads = [m.grad.detach() for m in self.masters if m.grad is not None]
        if not grads:
            return 0.0
        total = torch.zeros((), dtype=torch.float32, device=grads[0].device)
        for g in grads:
            total += g.float().pow(2).sum()
        return float(total.sqrt().cpu())

    def step_staged(self, max_norm: float | None = 1.0) -> None:
        """Clip the already-staged master grads and step. Pairs with `stage_grads_and_norm`."""
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.masters, max_norm, foreach=self.foreach)
        self.inner.step()
        with torch.no_grad():
            for p, m in zip(self.params, self.masters, strict=True):
                p.copy_(m.to(p.dtype))
                m.grad = None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            p.grad = None
        for m in self.masters:
            m.grad = None

    def state_dict(self) -> dict:
        return {"inner": self.inner.state_dict(),
                "masters": [m.detach().cpu() for m in self.masters]}

    def load_state_dict(self, sd: dict) -> None:
        self.inner.load_state_dict(sd["inner"])
        with torch.no_grad():
            for m, saved in zip(self.masters, sd["masters"], strict=True):
                m.copy_(saved.to(m.device))
            for p, m in zip(self.params, self.masters, strict=True):
                p.copy_(m.to(p.dtype))

    def load_masters(self, tensors: Sequence[torch.Tensor]) -> None:
        """Overwrite the masters (e.g. from a --resume latents payload) and
        propagate to the live params."""
        with torch.no_grad():
            for m, t in zip(self.masters, tensors, strict=True):
                m.copy_(t.to(m.device, torch.float32))
            for p, m in zip(self.params, self.masters, strict=True):
                p.copy_(m.to(p.dtype))
