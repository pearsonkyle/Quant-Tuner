"""Online on-policy distillation with a live acceptance curve.

Wires on-policy generation into the training loop so you can (a) generate only as
much target data as training actually consumes, and (b) read off an
acceptance-vs-steps curve to know how long to train for a target acceptance.

Because the target is FIXED, its greedy outputs are deterministic per prompt — so
this is not RL. That lets us decouple three things across the two GPUs:
  - a background PRODUCER thread hits the deployed target (e.g. :1234 on GPU 1)
    for greedy continuations and appends on-policy windows to a shared buffer;
  - the TRAINING loop (GPU 0) samples the buffer every step (never stalls once
    seeded — the fixed data is safe to revisit while the buffer grows);
  - EVAL runs in-process every ``eval_every`` steps on a held-out set, using the
    already-loaded target + current drafter, and logs ``(step, acceptance)``.

Training stops at ``max_steps``, on reaching ``target_acceptance``, or on a
plateau (no eval improvement over ``patience`` evals). The producer stops when
prompts are exhausted or training ends.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from quant_tuner.drafter.onpolicy import OnPolicyConfig, _generate


@dataclass
class OnlineConfig:
    target_model: str
    drafter_model: str
    out_dir: Path
    prompt_windows: Path
    eval_windows: Path
    gen_base_url: str = "http://127.0.0.1:1234/v1"
    gen_model: str = "gemma-4-e4b-w4a16-logs"
    seed_windows: Path | None = None
    """Optional pre-generated on-policy windows to warm the buffer at start."""

    prompt_len: int = 512
    gen_len: int = 512
    max_len: int = 1024
    lr: float = 1e-5
    grad_accum: int = 4
    warmup_steps: int = 20
    max_steps: int = 4000
    eval_every: int = 500
    eval_windows_n: int = 30
    target_acceptance: float | None = None
    """Stop early once eval acceptance reaches this (0-1)."""
    patience: int = 4
    """Stop if no eval improvement over this many evals."""
    gen_concurrency: int = 4
    max_gen: int = 4000
    load_target_4bit: bool = True
    device: str = "cuda:0"
    seed: int = 42
    ignore: tuple = field(default_factory=tuple)

    def validate(self) -> None:
        for p in (self.prompt_windows, self.eval_windows):
            if not Path(p).is_file():
                raise ValueError(f"file not found: {p}")
        if self.max_len < 2 or self.grad_accum < 1:
            raise ValueError("bad max_len/grad_accum")


class _Buffer:
    """Thread-safe growing pool of (ids, gen_start) windows."""

    def __init__(self) -> None:
        self._items: list[tuple[list[int], int]] = []
        self._lock = threading.Lock()

    def add(self, ids: list[int], gen_start: int) -> None:
        with self._lock:
            self._items.append((ids, gen_start))

    def sample(self, rng: random.Random) -> tuple[list[int], int] | None:
        with self._lock:
            if not self._items:
                return None
            return self._items[rng.randrange(len(self._items))]

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def _producer(cfg: OnlineConfig, buf: _Buffer, stop: threading.Event) -> None:
    """Generate on-policy windows into the buffer until prompts exhausted / stop."""
    op = OnPolicyConfig(
        base_url=cfg.gen_base_url, model=cfg.gen_model, out=Path("/dev/null"),
        prompt_windows=cfg.prompt_windows, prompt_len=cfg.prompt_len,
        gen_len=cfg.gen_len, max_prompts=cfg.max_gen, concurrency=cfg.gen_concurrency,
    )
    from concurrent.futures import ThreadPoolExecutor

    prompts: list[list[int]] = []
    with open(cfg.prompt_windows, encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= cfg.max_gen:
                break
            ids = json.loads(line)["input_ids"]
            if len(ids) >= cfg.prompt_len + 8:
                prompts.append(ids[: cfg.prompt_len])

    with ThreadPoolExecutor(cfg.gen_concurrency) as ex:
        futures = [ex.submit(_generate, op, p) for p in prompts]
        for fut in futures:
            if stop.is_set():
                break
            w = fut.result()
            if w and w["input_ids"][w["gen_start"]:]:
                buf.add(w["input_ids"][: cfg.max_len], w["gen_start"])


def _acceptance(target, assistant, eval_chunks, torch, device) -> float:
    from quant_tuner.drafter.train import _teacher_step  # noqa: F401

    assistant.eval()
    matched = total = 0
    with torch.no_grad():
        for ids, gen_start in eval_chunks:
            input_ids = torch.tensor([ids], device=device)
            tgt = target.model(input_ids=input_ids, return_shared_kv_states=True,
                               output_hidden_states=True, use_cache=False)
            emb = torch.cat([target.get_input_embeddings()(input_ids), tgt.hidden_states[-1]], dim=-1)
            logits = assistant(inputs_embeds=emb, shared_kv_states=tgt.shared_kv_states,
                               position_ids=torch.arange(input_ids.shape[1], device=device)[None]).logits
            lo = max(0, gen_start - 1)
            pred = logits[:, lo:-1].argmax(-1)
            lab = input_ids[:, lo + 1:]
            matched += (pred == lab).sum().item()
            total += lab.numel()
    assistant.train()
    return matched / max(1, total)


def train_online(cfg: OnlineConfig) -> Path:
    """Run online on-policy distillation. Writes the drafter + acceptance curve."""
    cfg.validate()
    import torch
    from transformers import AutoModelForCausalLM, Gemma4AssistantForCausalLM

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    quant = {}
    if cfg.load_target_4bit:
        from transformers import BitsAndBytesConfig
        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    target = AutoModelForCausalLM.from_pretrained(
        cfg.target_model, dtype=torch.bfloat16, device_map={"": cfg.device}, **quant).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    assistant = Gemma4AssistantForCausalLM.from_pretrained(
        cfg.drafter_model, dtype=torch.bfloat16).to(cfg.device)
    assistant.train()
    opt = torch.optim.AdamW(assistant.parameters(), lr=cfg.lr, betas=(0.9, 0.95))

    # fixed held-out eval set
    from quant_tuner.drafter.train import _teacher_step, load_windows
    eval_chunks = load_windows(Path(cfg.eval_windows), cfg.max_len)[: cfg.eval_windows_n]

    # buffer + producer
    buf = _Buffer()
    if cfg.seed_windows and Path(cfg.seed_windows).is_file():
        for ids, gs in load_windows(Path(cfg.seed_windows), cfg.max_len):
            buf.add(ids, gs)
    stop = threading.Event()
    prod = threading.Thread(target=_producer, args=(cfg, buf, stop), daemon=True)
    prod.start()

    # wait for a minimal buffer
    while len(buf) < 8:
        time.sleep(1)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    curve: list[dict] = []
    best = -1.0
    stale = 0

    def lr_at(s: int) -> float:
        return cfg.lr * (s + 1) / cfg.warmup_steps if s < cfg.warmup_steps else cfg.lr

    step = micro = 0
    running = 0.0
    opt.zero_grad()
    while step < cfg.max_steps:
        sample = buf.sample(rng)
        if sample is None:
            time.sleep(0.5)
            continue
        ids, gen_start = sample
        input_ids = torch.tensor([ids], device=cfg.device)
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
            if step % 20 == 0:
                print(f"step {step}/{cfg.max_steps} loss {running/cfg.grad_accum/20:.4f} "
                      f"buf {len(buf)}", flush=True)
                running = 0.0
            if step % cfg.eval_every == 0 or step == 1:
                acc = _acceptance(target, assistant, eval_chunks, torch, cfg.device)
                curve.append({"step": step, "acceptance": acc, "buffer": len(buf)})
                Path(cfg.out_dir, "acceptance_curve.json").write_text(json.dumps(curve, indent=2))
                print(f"[eval] step {step}: acceptance {acc*100:.2f}%  (buffer {len(buf)})", flush=True)
                if acc > best + 1e-4:
                    best = acc
                    stale = 0
                    assistant.save_pretrained(str(cfg.out_dir))  # keep best
                else:
                    stale += 1
                if cfg.target_acceptance and acc >= cfg.target_acceptance:
                    print(f"[stop] reached target acceptance {acc*100:.2f}%", flush=True)
                    break
                if stale >= cfg.patience:
                    print(f"[stop] plateau after {stale} evals (best {best*100:.2f}%)", flush=True)
                    break

    stop.set()
    (Path(cfg.out_dir) / "online_train.json").write_text(json.dumps(
        {"steps": step, "best_acceptance": best, "curve": curve,
         "target_model": cfg.target_model, "drafter_model": cfg.drafter_model}, indent=2))
    return Path(cfg.out_dir)
