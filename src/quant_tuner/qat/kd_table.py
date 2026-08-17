"""Serve precomputed top-K teacher logprobs to the training loop.

`kd_precompute.precompute_topk` writes one flat row per LABELED position — `(win, pos,
idx[K], logp[K], tail)` — in window order. The trainer consumes them one window at a time,
so this turns that flat table into a per-window lookup and checks the two things that
would otherwise corrupt the loss silently:

* **the table was built from THIS corpus** (fingerprint), because a table built from a
  different pack still has plausible-looking windows and positions, and KD against another
  corpus's teacher distribution is wrong everywhere without erroring anywhere;
* **the stored positions match the ones the trainer selected**, because `masked_forward`
  recomputes `keep_idx` from the labels and any divergence would silently pair each
  student position with a different token's teacher distribution.

The table is held on CPU (~2.3 GB for our corpus at top-64) and sliced per window; a
window's slice is ~3.6 MB, so the transfer is irrelevant next to the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class KDWindow:
    """One window's teacher rows, aligned to the trainer's ``keep_idx`` order."""

    idx: torch.Tensor      # [K_pos, topk] int32 — teacher's top-K token ids
    logp: torch.Tensor     # [K_pos, topk] fp16 — teacher logprobs (unnormalized over K)
    tail: torch.Tensor     # [K_pos]       fp16 — teacher mass outside the top-K

    def to(self, device) -> KDWindow:
        return KDWindow(self.idx.to(device, non_blocking=True),
                        self.logp.to(device, non_blocking=True),
                        self.tail.to(device, non_blocking=True))

    def slice(self, lo: int, hi: int) -> KDWindow:
        return KDWindow(self.idx[lo:hi], self.logp[lo:hi], self.tail[lo:hi])

    def __len__(self) -> int:
        return int(self.idx.shape[0])


class KDTable:
    """Precomputed teacher distributions, indexed by window."""

    def __init__(self, payload: dict, *, corpus_fingerprint: str | None = None):
        self.topk = int(payload["topk"])
        self.kd_vocab = int(payload["kd_vocab"])
        self.teacher = payload.get("teacher")
        self.fingerprint = payload.get("corpus_fingerprint")
        if (corpus_fingerprint and self.fingerprint
                and corpus_fingerprint != self.fingerprint):
            raise ValueError(
                f"KD table was built from corpus {self.fingerprint} but training on "
                f"{corpus_fingerprint}. The window/position indices would still resolve, "
                f"so this would distil against the wrong teacher distribution at every "
                f"position without erroring. Re-run kd_precompute on this corpus.")

        win = payload["win"].long()
        self._pos = payload["pos"].long()
        self._idx = payload["idx"]
        self._logp = payload["logp"]
        self._tail = payload["tail"]

        # Rows are written in window order, so each window owns a contiguous slice.
        # searchsorted gives the boundaries in one pass instead of a mask per window.
        n_win = int(payload.get("n_windows") or (int(win[-1]) + 1 if win.numel() else 0))
        bounds = torch.searchsorted(win, torch.arange(n_win + 1, dtype=win.dtype))
        self._lo, self._hi = bounds[:-1], bounds[1:]
        self.n_windows = n_win
        self.n_positions = int(payload.get("n_positions") or win.numel())

    @classmethod
    def load(cls, path: str | Path, *, corpus_fingerprint: str | None = None) -> KDTable:
        payload = torch.load(Path(path), weights_only=False)
        return cls(payload, corpus_fingerprint=corpus_fingerprint)

    def has_window(self, w: int) -> bool:
        return 0 <= w < self.n_windows and int(self._hi[w]) > int(self._lo[w])

    def for_window(self, w: int, keep_idx: torch.Tensor) -> KDWindow:
        """Rows for window ``w``, verified against the trainer's own position selection.

        ``keep_idx`` is recomputed by ``masked_forward`` from the labels; the table stored
        the same quantity at precompute time. They must agree exactly — a mismatch means
        the table and the corpus disagree about which positions are supervised, and pairing
        them anyway would distil every position against another token's distribution.
        """
        lo, hi = int(self._lo[w]), int(self._hi[w])
        if hi <= lo:
            raise KeyError(f"KD table has no rows for window {w}")
        pos = self._pos[lo:hi]
        k = keep_idx.detach().to(pos.device, torch.long)
        if pos.numel() != k.numel() or not torch.equal(pos, k):
            raise ValueError(
                f"KD table/window mismatch at window {w}: table has {pos.numel()} "
                f"supervised positions, the trainer selected {k.numel()}"
                + ("" if pos.numel() != k.numel() else " (same count, different positions)")
                + ". The table and the corpus disagree about what is supervised.")
        return KDWindow(self._idx[lo:hi], self._logp[lo:hi], self._tail[lo:hi])

    def coverage(self) -> float:
        """Mean teacher probability mass captured by the stored top-K.

        Reported once at startup: a top-K that captures only e.g. 60% of the mass makes the
        KL a much weaker constraint than it looks, and that is a property of the teacher's
        entropy on this corpus, not something the trainer can detect later.
        """
        return float(1.0 - self._tail.float().exp().mean())

    def __repr__(self) -> str:
        return (f"KDTable(teacher={self.teacher!r}, topk={self.topk}, "
                f"positions={self.n_positions:,}, windows={self.n_windows}, "
                f"coverage={self.coverage():.3f})")
