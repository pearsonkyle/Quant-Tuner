"""Shared helpers for end-to-end experiment driver scripts.

Each calibration study (e.g. the OmniCoder-9B Q4_K_M leaderboard) is composed
of multiple stages that produce GGUFs, imatrix files, and CSV rows. These
helpers centralize the timing/log conventions and the idempotent
"build artifact X if missing" pattern those drivers share.
"""

from quant_tuner.experiments.runner import log, phase, step

__all__ = ["log", "phase", "step"]
