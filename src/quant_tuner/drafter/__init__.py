"""MTP-drafter fine-tuning on long agentic sessions.

Produces a fine-tuned Gemma-4 assistant (drafter) calibrated on OUR usage logs
at long context — a pure speed lever for the vLLM osoi5/turbo serving stack
(under greedy speculative decoding the drafter cannot change emitted tokens).

- ``windows``: long-context training windows from logs (importable, no torch).
- ``train``: the coupled teacher-forced loop (torch/transformers at call time).
"""

from quant_tuner.drafter.windows import WindowConfig, iter_windows, write_windows

__all__ = ["WindowConfig", "iter_windows", "write_windows"]
