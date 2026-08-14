"""Teacher-forced fine-tuning of the Gemma-4 MTP assistant (drafter).

The assistant is NOT a standalone LM: its ``forward`` needs ``inputs_embeds``
and ``shared_kv_states`` from the frozen target. So training is a coupled loop
(EAGLE/MTP-style):

    target(input_ids, return_shared_kv_states=True, output_hidden_states=True)
        -> hidden_states[0]  = scaled input embeddings   (assistant inputs_embeds)
        -> shared_kv_states  = {full_attention:(K,V), sliding_attention:(K,V)}
    assistant(inputs_embeds=embeds, shared_kv_states=kv) -> logits
    loss = CE(logits[:, :-1], input_ids[:, 1:])     # predict next token

Only the assistant gets gradients; the target is frozen (loaded 4-bit to fit a
16 GB card alongside long-context activations). Under greedy speculative
decoding the drafter can never change an emitted token — it only raises
acceptance — so this is a pure latency knob with no quality risk to the served
model. Calibrated on OUR long agentic sessions (median ~46k tokens), unlike the
challenge drafter tuned to a ≤4k benchmark.

Heavy deps (torch/transformers/bitsandbytes) import at call time; the module
imports without them so window/config logic stays unit-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    target_model: str
    """bf16 target checkpoint (google/gemma-4-E4B-it) — provides embeds + shared KV."""
    drafter_model: str
    """Warm-start assistant checkpoint (kenyan-duma ft, or google's base assistant)."""
    windows: Path
    """JSONL from drafter.windows (long-context training windows)."""
    out_dir: Path

    max_len: int = 8192
    """Per-step sequence cap. Windows longer than this are chunked so a 32k
    agentic window still trains, one max_len slice at a time, within 16 GB."""
    epochs: float = 1.0
    lr: float = 1e-4
    grad_accum: int = 4
    warmup_steps: int = 20
    load_target_4bit: bool = True
    """4-bit target (bitsandbytes) so target + long activations fit one 16 GB GPU.
    shared_kv_states quality is unaffected in practice for drafter distillation."""
    target_device: str = "cuda:0"
    drafter_device: str = "cuda:0"
    log_every: int = 10
    save_every: int = 200
    max_steps: int | None = None
    """Cap total optimizer steps (smoke runs). None = full epochs."""
    seed: int = 42
    ignore: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not Path(self.windows).is_file():
            raise ValueError(f"windows file not found: {self.windows}")
        if self.max_len < 2:
            raise ValueError("max_len must be >= 2")
        if self.grad_accum < 1:
            raise ValueError("grad_accum must be >= 1")


def load_windows(path: Path, max_len: int) -> list[tuple[list[int], int]]:
    """Read windows JSONL as (ids, gen_start) pairs. ``gen_start`` marks the first
    on-policy (target-generated) position — loss is computed only from there on, so
    the drafter learns the target's own outputs, not the prompt it was fed. Windows
    without ``gen_start`` train on all positions (gen_start=0) and long ones are
    sliced into <= max_len chunks. On-policy windows (prompt+gen, already <= max_len)
    are kept whole so gen_start stays aligned."""
    chunks: list[tuple[list[int], int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids = rec["input_ids"]
            gs = rec.get("gen_start")
            if gs is not None:
                ids = ids[:max_len]
                if len(ids) >= 2 and gs < len(ids):
                    chunks.append((ids, gs))
                continue
            for start in range(0, len(ids), max_len):
                piece = ids[start : start + max_len]
                if len(piece) >= 2:
                    chunks.append((piece, 0))
    return chunks


def _teacher_step(target, assistant, input_ids, torch, gen_start=0):
    """One coupled forward. Returns (loss, n_predicted_tokens).

    ``gen_start`` (on-policy): compute loss only on target-generated positions
    (index >= gen_start), i.e. predicting token t+1 for t >= gen_start-1.

    The assistant's EAGLE-style input at position t is
    ``concat(raw_embed(token_t), target_hidden_state_t)`` (2*hidden = 5120), and
    it predicts token t+1 (see transformers' AssistedCandidateGeneratorShared).
    Teacher-forced, this vectorizes over the whole sequence in one pass.
    """
    with torch.no_grad():
        tgt_out = target.model(
            input_ids=input_ids,
            return_shared_kv_states=True,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = tgt_out.hidden_states[-1]  # h_t, causal (attends <= t)
        shared_kv = tgt_out.shared_kv_states
        raw_embed = target.get_input_embeddings()(input_ids)  # unscaled, per inference path
        inputs_embeds = torch.cat([raw_embed, last_hidden], dim=-1)  # [1, T, 2*hidden]

    asst_out = assistant(
        inputs_embeds=inputs_embeds,
        shared_kv_states=shared_kv,
        position_ids=torch.arange(input_ids.shape[1], device=input_ids.device)[None],
    )
    logits = asst_out.logits  # [1, T, vocab] — vocab is 262k, so avoid a single
    # fp32 materialization of the whole thing; chunk CE over the sequence.
    # predicting token t+1 sits at logit index t; mask to on-policy positions.
    lo = max(0, gen_start - 1)
    shift_logits = logits[:, lo:-1].reshape(-1, logits.shape[-1])
    shift_labels = input_ids[:, lo + 1 :].reshape(-1)
    n = shift_labels.numel()
    chunk = 512
    total = shift_logits.new_zeros(())
    for i in range(0, n, chunk):
        sl = shift_logits[i : i + chunk].float()
        total = total + torch.nn.functional.cross_entropy(
            sl, shift_labels[i : i + chunk], reduction="sum"
        )
    return total / max(1, n), n


def train(cfg: TrainConfig) -> Path:
    """Run the coupled fine-tune. Returns the saved drafter dir.

    Requires the ``drafter`` extra (torch + transformers + bitsandbytes)."""
    cfg.validate()
    import torch
    from transformers import (
        AutoModelForCausalLM,
        Gemma4AssistantForCausalLM,
    )

    torch.manual_seed(cfg.seed)

    quant_kwargs: dict[str, Any] = {}
    if cfg.load_target_4bit:
        from transformers import BitsAndBytesConfig

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    target = AutoModelForCausalLM.from_pretrained(
        cfg.target_model,
        torch_dtype=torch.bfloat16,
        device_map={"": cfg.target_device},
        **quant_kwargs,
    )
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    assistant = Gemma4AssistantForCausalLM.from_pretrained(
        cfg.drafter_model, torch_dtype=torch.bfloat16
    ).to(cfg.drafter_device)
    assistant.train()

    opt = torch.optim.AdamW(assistant.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    chunks = load_windows(Path(cfg.windows), cfg.max_len)
    total_steps = cfg.max_steps or int(len(chunks) * cfg.epochs / cfg.grad_accum)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        return cfg.lr

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    step = 0
    micro = 0
    opt.zero_grad()
    running = 0.0
    for _epoch in range(max(1, int(cfg.epochs + 0.999))):
        for ids, gen_start in chunks:
            input_ids = torch.tensor([ids], device=cfg.target_device)
            loss, _ = _teacher_step(target, assistant, input_ids, torch, gen_start=gen_start)
            (loss / cfg.grad_accum).backward()
            running += loss.item()
            micro += 1
            if micro % cfg.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(assistant.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % cfg.log_every == 0:
                    print(
                        f"step {step}/{total_steps} loss {running / cfg.grad_accum / cfg.log_every:.4f} "
                        f"lr {lr_at(step):.2e}",
                        flush=True,
                    )
                    running = 0.0
                if cfg.save_every and step % cfg.save_every == 0:
                    assistant.save_pretrained(str(Path(cfg.out_dir) / f"step-{step}"))
                if cfg.max_steps and step >= cfg.max_steps:
                    break
        if cfg.max_steps and step >= cfg.max_steps:
            break

    assistant.save_pretrained(str(cfg.out_dir))
    (Path(cfg.out_dir) / "drafter_train.json").write_text(
        json.dumps(
            {
                "target_model": cfg.target_model,
                "drafter_model": cfg.drafter_model,
                "windows": str(cfg.windows),
                "steps": step,
                "max_len": cfg.max_len,
                "lr": cfg.lr,
            },
            indent=2,
        )
    )
    return Path(cfg.out_dir)
