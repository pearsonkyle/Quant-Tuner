#!/usr/bin/env python
"""Run MMLU-Pro on multiple models with progress monitoring."""
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from quant_tuner.eval.mmlu_pro import (
    run_mmlu_pro_eval,
    MmluProHoldout,
    Sampling,
)
from openai import OpenAI

MODEL_PATHS = [
    Path("out/exp-001/Jackrong__Qwopus3.5-9B-Coder/model-f16.gguf"),
    Path("out/exp-001/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-none.gguf"),
    Path("out/exp-002/Jackrong__Qwopus3.5-9B-Coder/Q4_K_M-outlier_l4.gguf"),
]

SAMPLING = Sampling(temperature=0.6, top_p=0.95, top_k=20)
HOLDOUT_PATH = Path("out/mmlu_pro/holdout.json")
LOG_DIR = Path("out/mmlu_pro/logs")

for model_path in MODEL_PATHS:
    label = model_path.stem
    print(f"\n{'='*60}")
    print(f"Model: {model_path.name} ({label})")
    print(f"{'='*60}")
    try:
        summary = run_mmlu_pro_eval(
            holdout_path=HOLDOUT_PATH,
            model_path=str(model_path),
            sampling=SAMPLING,
            ctx=8192,
            ngl=99,
            per_sample_log=f"{LOG_DIR}/{label}.jsonl",
            progress=True,
        )
        print(f"\n{summary.render_summary()}")
    except Exception as e:
        print(f"ERROR for {model_path}: {e}")