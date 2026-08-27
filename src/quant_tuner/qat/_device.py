"""Backend abstraction for the QAT trainer: MPS, CUDA and CPU in one object.

Everything in this package was written on an M4 Max, so a pile of decisions that read
like general engineering are really Metal workarounds. Left as bare ``if dev == "mps"``
branches they are invisible — a CUDA box inherits the workaround, or (worse) silently
misses the branch entirely and lands on CPU. This module names each one, states the
device it exists for, and gives the trainer a single object to ask.

The differences that actually matter, and why:

``foreach``
    MPS multi-tensor (foreach) optimizer and ``clip_grad_norm_`` kernels deadlock at
    full-model scale, so every loop here is per-tensor. On CUDA the foreach path is a
    real speedup over ~250 separate small kernel launches, and there is no deadlock.

``needs_chunked_sdpa``
    Metal has no fused SDPA training kernel: the backward pass materializes
    ``[n_heads, S, S]`` and MPSGraph refuses a tensor with more than INT_MAX elements,
    capping the window at ``n_heads * S^2 < 2^31`` (S <= 8191 at 32 heads). CUDA has
    FlashAttention, which never materializes the score matrix at all — so the chunked
    path is unnecessary there and probably slower. ``qat.attention``'s chunking stays
    available on every device because ``--trained-tail`` needs it to carry the prefix
    K/V, but it is only *forced* where the hardware demands it.

``teacher_dtype``
    KD teacher precision. fp16 on MPS (no bf16 on M1-generation parts, and fp16 is the
    well-trodden Metal path), bf16 on CUDA, fp32 on CPU.

``empty_cache_every``
    On MPS the allocator fragments over a long all-36 run until the working set creeps
    into swap and macOS SIGKILLs the process, so the cache is released every few steps.
    CUDA's caching allocator has no such failure mode; ``empty_cache`` there costs a
    device synchronize and hands back blocks the allocator would have reused. Default
    it off and let the user opt in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quant_tuner.calibrate._device import resolve_device

#: MPSGraph refuses a tensor with > INT_MAX elements, and the unfused Metal training SDPA
#: path materializes [n_heads, S, S]. The ceiling is n_heads*S^2 < 2^31, i.e. S <= 8191 at
#: 32 heads — 8192 fails by exactly ONE element (32*8192^2 == 2^31). Measured fwd+bwd on
#: torch 2.12/M4 Max: 4096/6144/7168/8064/8128/8191 all pass, 8192 is the only failure.
MPS_MAX_WINDOW = 8191


@dataclass(frozen=True)
class Backend:
    """One accelerator's capabilities and quirks, resolved once at trainer startup."""

    name: str
    #: multi-tensor optimizer / clip kernels safe to use?
    foreach: bool
    #: must SDPA be query-chunked for a long window to train at all?
    needs_chunked_sdpa: bool
    #: hard window ceiling WITHOUT chunked attention (None = memory is the only limit)
    max_window: int | None
    #: dtype for a KD teacher held alongside the student
    teacher_dtype: torch.dtype
    #: default ``--empty-cache-every``; 0 disables
    default_empty_cache_every: int

    @property
    def is_cuda(self) -> bool:
        return self.name.startswith("cuda")

    @property
    def is_mps(self) -> bool:
        return self.name == "mps"

    def empty_cache(self) -> None:
        """Return cached-but-free blocks to the driver. No-op on CPU."""
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.is_mps:
            torch.mps.empty_cache()

    def allocated_gib(self) -> float:
        """Live tensor bytes, in GiB. 0.0 where the backend cannot report it."""
        if self.is_cuda:
            return torch.cuda.memory_allocated(self.name) / 1024**3
        if self.is_mps:
            return torch.mps.current_allocated_memory() / 1024**3
        return 0.0

    def peak_gib(self) -> float:
        """High-water mark since the last :meth:`reset_peak`, in GiB.

        This is the number a run is sized from — the transient a checkpoint save or a
        long-window validation spikes to is what decides whether the next one OOMs, and
        a point sample of live bytes will not show it. MPS exposes no peak counter, so
        it falls back to the current allocation there.
        """
        if self.is_cuda:
            return torch.cuda.max_memory_allocated(self.name) / 1024**3
        return self.allocated_gib()

    def reset_peak(self) -> None:
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.name)

    def total_gib(self) -> float:
        """Physical capacity of the device, in GiB (0.0 when unknown, e.g. CPU/MPS)."""
        if self.is_cuda:
            return torch.cuda.get_device_properties(self.name).total_memory / 1024**3
        return 0.0

    def describe(self) -> str:
        if self.is_cuda:
            p = torch.cuda.get_device_properties(self.name)
            return (f"{self.name} ({p.name}, {p.total_memory / 1024**3:.0f} GiB, "
                    f"cc {p.major}.{p.minor}, {torch.cuda.device_count()} visible)")
        return self.name


_SPECS = {
    "cuda": dict(foreach=True, needs_chunked_sdpa=False, max_window=None,
                 teacher_dtype=torch.bfloat16, default_empty_cache_every=0),
    "mps": dict(foreach=False, needs_chunked_sdpa=True, max_window=MPS_MAX_WINDOW,
                teacher_dtype=torch.float16, default_empty_cache_every=5),
    "cpu": dict(foreach=True, needs_chunked_sdpa=False, max_window=None,
                teacher_dtype=torch.float32, default_empty_cache_every=0),
}


def resolve_backend(device: str = "auto") -> Backend:
    """Resolve ``"auto"`` (cuda > mps > cpu) or an explicit device string to a Backend.

    An explicit ``cuda:1`` keeps its index — the memory counters are per-device, so the
    reported numbers follow the device the model is actually on.
    """
    name = resolve_device(device)
    kind = "cuda" if name.startswith("cuda") else name
    if kind not in _SPECS:
        raise ValueError(f"unsupported QAT device {name!r}; expected cuda/mps/cpu")
    return Backend(name=name, **_SPECS[kind])  # type: ignore[arg-type]
