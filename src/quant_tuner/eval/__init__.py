"""Task-level evaluation: tool-call accuracy on held-out sessions.

KLD measures token-level fidelity but doesn't tell you whether a quant still
emits *valid tool calls* on real agentic traffic. This package runs each quant
against a holdout corpus via llama-server's OpenAI-compatible endpoint and
scores tool selection, parameter accuracy, schema validity, and rollout
completion. See ``docs/benchmarks.md`` for metric definitions.
"""

from quant_tuner.eval.mmlu_pro import (
    MmluProHoldout,
    MmluProSample,
    MmluProSummary,
    run_mmlu_pro_eval,
)
from quant_tuner.eval.scoring import (
    compare_value,
    is_schema_valid,
    param_score,
    parse_arguments,
    schema_for,
)
from quant_tuner.eval.toolcall import EvalSummary, run_toolcall_eval

__all__ = [
    "EvalSummary",
    "MmluProHoldout",
    "MmluProSample",
    "MmluProSummary",
    "compare_value",
    "is_schema_valid",
    "param_score",
    "parse_arguments",
    "run_mmlu_pro_eval",
    "run_toolcall_eval",
    "schema_for",
]
