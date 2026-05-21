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
