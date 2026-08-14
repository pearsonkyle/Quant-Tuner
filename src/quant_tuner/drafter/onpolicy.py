"""On-policy distillation data for MTP-drafter training.

The drafter's only job is to predict *what the target actually emits*. Training it
on human logs or another model's text (off-policy) is why MTP-alone tuning has been
flat — the drafter never sees the target's own output distribution. Standard
EAGLE/MTP practice is on-policy: have the TARGET greedily generate continuations,
and train the drafter to predict *those* tokens.

This module queries a deployed OpenAI-compatible target (e.g. the HomeLab W4A16
server on :1234) over prompts derived from our windows, captures the exact
generated token ids (`return_token_ids: true`), and writes training windows of
``prompt_ids + target_generated_ids`` with a ``gen_start`` marker so the trainer
can mask loss to the on-policy (generated) span.

Deterministic given (prompts, greedy decoding). Concurrency via a thread pool —
the server does the work; we just fan out requests.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OnPolicyConfig:
    base_url: str
    """OpenAI-compatible base, e.g. http://127.0.0.1:1234/v1"""
    model: str
    out: Path
    prompt_windows: Path
    """Windows JSONL whose input_ids provide the prompt prefixes."""
    prompt_len: int = 512
    """Take this many leading tokens of each window as the prompt."""
    gen_len: int = 512
    """max_tokens to greedily generate from the target."""
    max_prompts: int = 4000
    concurrency: int = 8
    timeout_s: int = 180

    def validate(self) -> None:
        if not Path(self.prompt_windows).is_file():
            raise ValueError(f"prompt_windows not found: {self.prompt_windows}")
        if self.prompt_len < 1 or self.gen_len < 1:
            raise ValueError("prompt_len and gen_len must be >= 1")


def _iter_prompts(cfg: OnPolicyConfig) -> Iterator[list[int]]:
    n = 0
    with open(cfg.prompt_windows, encoding="utf-8") as f:
        for line in f:
            if n >= cfg.max_prompts:
                return
            ids = json.loads(line)["input_ids"]
            if len(ids) < cfg.prompt_len + 8:  # need room for a real continuation
                continue
            yield ids[: cfg.prompt_len]
            n += 1


def _generate(cfg: OnPolicyConfig, prompt_ids: list[int]) -> dict | None:
    """One greedy generation. Returns a window dict or None on failure."""
    body = json.dumps({
        "model": cfg.model,
        "prompt": prompt_ids,
        "max_tokens": cfg.gen_len,
        "temperature": 0,
        "return_token_ids": True,
    }).encode()
    req = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/completions", body, {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
            choice = json.load(r)["choices"][0]
    except Exception:
        return None
    gen_ids = choice.get("token_ids")
    if not gen_ids:
        return None
    # exact on-policy sequence: prompt the target saw + tokens it emitted
    input_ids = list(prompt_ids) + list(gen_ids)
    return {
        "input_ids": input_ids,
        "gen_start": len(prompt_ids),
        "source": "onpolicy",
        "n_tokens": len(input_ids),
    }


def generate_windows(cfg: OnPolicyConfig) -> dict:
    """Fan out greedy generations and write on-policy windows JSONL."""
    cfg.validate()
    prompts = list(_iter_prompts(cfg))
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    written = gen_tokens = 0
    with open(cfg.out, "w", encoding="utf-8") as out, ThreadPoolExecutor(cfg.concurrency) as ex:
        futures = [ex.submit(_generate, cfg, p) for p in prompts]
        for fut in as_completed(futures):
            w = fut.result()
            if w is None:
                continue
            out.write(json.dumps(w) + "\n")
            out.flush()
            written += 1
            gen_tokens += w["n_tokens"] - w["gen_start"]
            if written % 100 == 0:
                print(f"[onpolicy] {written}/{len(prompts)} generations, "
                      f"{gen_tokens/1e6:.1f}M generated tokens", flush=True)
    stats = {"prompts": len(prompts), "windows": written, "generated_tokens": gen_tokens}
    cfg.out.with_suffix(".audit.json").write_text(json.dumps(stats, indent=2))
    return stats
