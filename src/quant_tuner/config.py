"""Recipe + run configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Method = Literal["imatrix", "awq", "gptq", "none"]


class DataConfig(BaseModel):
    logs: Path | None = None
    corpus: Path | None = None
    supplement: Path | None = None
    train_frac: float = 0.8
    test_frac: float = 0.1
    holdout_frac: float = 0.1
    train_max_tokens: int = 250_000
    eval_max_tokens: int = 50_000
    context_len: int = 16_384
    # Calibration-corpus packing. With system_prose_budget set (the default),
    # the calibration (train) corpus is built with the stub + multi-window
    # packer: each session is sliced into <=per_session_cap windows whose system
    # prose is trimmed to system_prose_budget tokens, and the full prose is
    # rendered in only full_prose_quota sessions per unique system prompt.
    # Tool schemas get the same dedup via tool_schema_quota: the full schema
    # set renders in the first window of the first `quota` sessions per unique
    # set, a name+description stub everywhere else (null disables — pre-fix
    # behavior re-rendered full schemas in EVERY window). Keep per_session_cap
    # < the imatrix ctx (default 4096) so no window straddles a context
    # boundary. Set system_prose_budget: null to fall back to the legacy
    # head-truncated single-chunk packer.
    per_session_cap: int = 3_500
    system_prose_budget: int | None = 256
    full_prose_quota: int = 1
    max_windows_per_session: int = 8
    tool_schema_quota: int | None = 1


class CalibrationConfig(BaseModel):
    method: Method = "imatrix"
    variant: str = "default"
    # Method-specific overrides; calibrators read what they need.
    params: dict = Field(default_factory=dict)


class ExtractConfig(BaseModel):
    keep_mtp: bool = False


class QuantizeConfig(BaseModel):
    type: str = "Q4_K_M"  # any llama-quantize tag


class BenchConfig(BaseModel):
    suite: Literal["quick", "kld", "speed", "full", "leaderboard"] = "full"
    reference: Path | None = None
    eval_ctx: int = 8192
    # Pre-built PPL/KLD eval corpus (e.g. scripts/build_corpora.py's external
    # corpus.eval.txt). Strongly preferred over the default log-derived eval:
    # llama-perplexity cannot --parse-special, so a chat-templated eval slice
    # is tokenized differently from what the model sees at inference.
    eval_corpus: Path | None = None
    # Optional second, in-distribution KLD suite (scripts/build_corpora.py's
    # corpus.eval.tools.txt — windowed logtrain holdout). Gets its own
    # baseline (eval/baseline-tools.kld) and emits *_tools columns. Chat
    # markers tokenize as plain BPE (no --parse-special), so the numbers are
    # quant-vs-quant comparisons, not absolute PPL.
    eval_tools_corpus: Path | None = None


class RunConfig(BaseModel):
    name: str
    model: str  # HF repo id or local path
    workspace: Path
    data: DataConfig = Field(default_factory=DataConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    quantize: QuantizeConfig = Field(default_factory=QuantizeConfig)
    bench: BenchConfig = Field(default_factory=BenchConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
