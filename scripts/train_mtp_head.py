"""Fine-tune the MTP (multi-token prediction) head for a Qwen3.5-9B-family model.

The backbone (main transformer + embeddings + lm_head) is fully frozen.
Only the mtp.* parameters are trained.

Forward pass matching llama.cpp qwen35.cpp graph_mtp:

    h_norm  = RMSNorm(hidden_t,        pre_fc_norm_hidden)
    e_norm  = RMSNorm(embed(x_{t+1}),  pre_fc_norm_embedding)
    cur     = fc( cat(h_norm, e_norm) )   # [2*hidden -> hidden]
    cur     = mtp_block(cur)              # full-attention decoder layer
    cur     = RMSNorm(cur, norm)
    logits  = lm_head(cur)                # predicts x_{t+2}

The loss is cross-entropy between logits at positions [0, T-3] and
tokens at positions [2, T-1]  (i.e. a 2-step-ahead prediction).

Usage::

    # Typical run against the tesslate model
    uv run python scripts/train_mtp_head.py \\
        --model out/benchmark_9b_iq3s/tesslate/model_extracted \\
        --logs logtrain.jsonl \\
        --out out/benchmark_9b_iq3s/tesslate/mtp_trained \\
        --steps 500 --lr 2e-5

    # Dry run — prints plan and exits
    uv run python scripts/train_mtp_head.py \\
        --model out/benchmark_9b_iq3s/tesslate/model_extracted \\
        --logs logtrain.jsonl \\
        --out out/benchmark_9b_iq3s/tesslate/mtp_trained \\
        --dry-run

After training, copy the saved mtp weights into model_extracted and
reconvert to GGUF, then run bench_mtp_speed.py with --holdout-jsonl.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Default relative activation-noise levels per quant type.
# Measured empirically as ||h_fp16 - h_quant||_F / ||h_fp16||_F at the final
# hidden layer on Qwen3-family models.  Run scripts/measure_quant_noise.py to
# calibrate for a specific GGUF; override with --quant-noise.
QUANT_NOISE_DEFAULTS: dict[str, float] = {
    "Q8_0":   0.002,
    "Q6_K":   0.005,
    "Q5_K_S": 0.010,
    "Q5_K_M": 0.013,
    "Q4_K_S": 0.020,
    "Q4_K_M": 0.022,
    "Q3_K_M": 0.048,
    "IQ4_XS": 0.018,
    "IQ4_NL": 0.016,
    "IQ3_S":  0.058,
    "IQ3_M":  0.052,
    "IQ2_S":  0.115,
    "IQ2_M":  0.095,
}

from quant_tuner.data import ingest, split
from quant_tuner.experiments import log, phase
from quant_tuner.mtp.chunks import build_token_batches
from quant_tuner.mtp.chunks import session_to_text as _session_to_text

# ---------------------------------------------------------------------------
# MTP head
# ---------------------------------------------------------------------------

def _load_mtp_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all mtp.* tensors from model_extracted safetensors."""
    import json

    from safetensors.torch import load_file

    index_path = model_dir / "model.safetensors.index.json"
    wmap: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
    mtp_shards: dict[str, list[str]] = {}
    for key, shard in wmap.items():
        if key.startswith("mtp."):
            mtp_shards.setdefault(shard, []).append(key)

    tensors: dict[str, torch.Tensor] = {}
    for shard, keys in mtp_shards.items():
        loaded = load_file(str(model_dir / shard))
        tensors.update({k: loaded[k] for k in keys})
    return tensors


class Qwen3MTPBlock(torch.nn.Module):
    """Dense-attention-only transformer block for the MTP head.

    Architecture matches blk.32 in Qwen3.5-9B GGUF (full_attention type).
    """

    def __init__(self, config: object) -> None:
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
        )
        # Build a fresh config copy that forces full_attention for this layer.
        mtp_config = copy.deepcopy(config)
        # layer_types must have at least one entry at index 0
        mtp_config.layer_types = ["full_attention"]
        self.layer = Qwen3_5DecoderLayer(mtp_config, layer_idx=0)
        self.hidden_size = config.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


class Qwen3MTPHead(torch.nn.Module):
    """Full Qwen3.5 MTP head: fc + mtp_block + norm.

    The lm_head is NOT included here; it is the backbone's lm_head (shared).
    """

    def __init__(self, config: object) -> None:
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        h = config.hidden_size
        self.pre_fc_norm_hidden     = Qwen3_5RMSNorm(h, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding  = Qwen3_5RMSNorm(h, eps=config.rms_norm_eps)
        self.fc                     = torch.nn.Linear(2 * h, h, bias=False)
        self.block                  = Qwen3MTPBlock(config)
        self.norm                   = Qwen3_5RMSNorm(h, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, T, H] from frozen backbone
        next_token_embeds: torch.Tensor,       # [B, T, H] embed(x_{t+1})
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_norm = self.pre_fc_norm_hidden(hidden_states)
        e_norm = self.pre_fc_norm_embedding(next_token_embeds)
        cur    = self.fc(torch.cat([h_norm, e_norm], dim=-1))
        cur    = self.block(cur, position_embeddings, attention_mask, position_ids)
        cur    = self.norm(cur)
        return cur  # caller applies lm_head

    def load_from_safetensors(self, tensors: dict[str, torch.Tensor]) -> None:
        """Load mtp.* tensors into this module."""
        sd = self.state_dict()
        mapping = {
            "pre_fc_norm_hidden.weight":    "mtp.pre_fc_norm_hidden.weight",
            "pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding.weight",
            "fc.weight":                    "mtp.fc.weight",
            "norm.weight":                  "mtp.norm.weight",
            # block layer
            "block.layer.input_layernorm.weight":         "mtp.layers.0.input_layernorm.weight",
            "block.layer.post_attention_layernorm.weight":"mtp.layers.0.post_attention_layernorm.weight",
            "block.layer.self_attn.q_proj.weight":        "mtp.layers.0.self_attn.q_proj.weight",
            "block.layer.self_attn.k_proj.weight":        "mtp.layers.0.self_attn.k_proj.weight",
            "block.layer.self_attn.v_proj.weight":        "mtp.layers.0.self_attn.v_proj.weight",
            "block.layer.self_attn.o_proj.weight":        "mtp.layers.0.self_attn.o_proj.weight",
            "block.layer.self_attn.q_norm.weight":        "mtp.layers.0.self_attn.q_norm.weight",
            "block.layer.self_attn.k_norm.weight":        "mtp.layers.0.self_attn.k_norm.weight",
            "block.layer.mlp.gate_proj.weight":           "mtp.layers.0.mlp.gate_proj.weight",
            "block.layer.mlp.up_proj.weight":             "mtp.layers.0.mlp.up_proj.weight",
            "block.layer.mlp.down_proj.weight":           "mtp.layers.0.mlp.down_proj.weight",
        }
        new_sd = {k: tensors[v].to(sd[k].dtype) for k, v in mapping.items() if v in tensors}
        missing = [k for k, v in mapping.items() if v not in tensors]
        if missing:
            log(f"  [warn] missing mtp tensors: {missing}")
        self.load_state_dict(new_sd, strict=False)

    def to_safetensors_dict(self) -> dict[str, torch.Tensor]:
        """Export state dict back to mtp.* naming for saving."""
        sd = self.state_dict()
        reverse_map = {
            "pre_fc_norm_hidden.weight":                  "mtp.pre_fc_norm_hidden.weight",
            "pre_fc_norm_embedding.weight":               "mtp.pre_fc_norm_embedding.weight",
            "fc.weight":                                  "mtp.fc.weight",
            "norm.weight":                                "mtp.norm.weight",
            "block.layer.input_layernorm.weight":         "mtp.layers.0.input_layernorm.weight",
            "block.layer.post_attention_layernorm.weight":"mtp.layers.0.post_attention_layernorm.weight",
            "block.layer.self_attn.q_proj.weight":        "mtp.layers.0.self_attn.q_proj.weight",
            "block.layer.self_attn.k_proj.weight":        "mtp.layers.0.self_attn.k_proj.weight",
            "block.layer.self_attn.v_proj.weight":        "mtp.layers.0.self_attn.v_proj.weight",
            "block.layer.self_attn.o_proj.weight":        "mtp.layers.0.self_attn.o_proj.weight",
            "block.layer.self_attn.q_norm.weight":        "mtp.layers.0.self_attn.q_norm.weight",
            "block.layer.self_attn.k_norm.weight":        "mtp.layers.0.self_attn.k_norm.weight",
            "block.layer.mlp.gate_proj.weight":           "mtp.layers.0.mlp.gate_proj.weight",
            "block.layer.mlp.up_proj.weight":             "mtp.layers.0.mlp.up_proj.weight",
            "block.layer.mlp.down_proj.weight":           "mtp.layers.0.mlp.down_proj.weight",
        }
        return {hf_name: sd[local].cpu().to(torch.bfloat16)
                for local, hf_name in reverse_map.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Holdout evaluation (MTP perplexity on the held-out text split)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_mtp_ppl(
    backbone: object,
    mtp_head: Qwen3MTPHead,
    device: torch.device,
    model_dir: Path,
    logtrain: Path,
    seq_len: int,
    max_tokens: int = 20_000,
) -> tuple[float, float]:
    """Compute MTP perplexity and top-1 accuracy on the logtrain holdout text split.

    Returns (ppl, top1_accuracy). top1_accuracy is the fraction of positions where
    argmax(MTP logits) matches the ground-truth next-next token — a proxy for
    speculative-decoding acceptance rate under greedy/near-greedy sampling.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), fix_mistral_regex=True)
    sessions = ingest.load_sessions(logtrain)
    sessions = ingest.filter_sessions(sessions, min_score=0.3, require_tools=False)
    splits = split.split_sessions(
        sessions, train_frac=0.8, test_frac=0.1, holdout_frac=0.1, seed=42
    )
    holdout_sessions = splits["holdout"]
    all_ids: list[int] = []
    for s in holdout_sessions:
        text = _session_to_text(s, tok)
        ids = tok.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        if len(all_ids) >= max_tokens:
            break
    all_ids = all_ids[:max_tokens]

    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0

    for i in range(0, len(all_ids) - seq_len - 2, seq_len):
        chunk = all_ids[i : i + seq_len + 2]
        input_ids = torch.tensor([chunk[:-2]], dtype=torch.long, device=device)
        next_ids  = torch.tensor([chunk[1:-1]], dtype=torch.long, device=device)
        labels    = torch.tensor([chunk[2:]], dtype=torch.long, device=device)

        out = backbone(input_ids=input_ids, output_hidden_states=True)
        h   = out.hidden_states[-1]                     # [1, T, H]
        e   = backbone.model.embed_tokens(next_ids)     # [1, T, H]

        B, T, _ = h.shape
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        rotary  = backbone.model.rotary_emb(h, pos_ids)

        mtp_out = mtp_head(h, e, rotary, position_ids=pos_ids)
        logits  = backbone.lm_head(mtp_out)             # [1, T, V]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )
        total_loss   += loss.item()
        total_tokens += labels.numel()

        # Top-1: how often does argmax(MTP) == ground truth?
        preds = logits.argmax(dim=-1)                   # [1, T]
        top1_correct += (preds == labels).sum().item()

    ppl = math.exp(total_loss / max(total_tokens, 1))
    top1_acc = top1_correct / max(total_tokens, 1)
    return ppl, top1_acc


# ---------------------------------------------------------------------------
# Save trained weights
# ---------------------------------------------------------------------------

def _save_mtp_weights(
    mtp_head: Qwen3MTPHead,
    model_dir: Path,
    out_dir: Path,
) -> None:
    """Save trained mtp.* weights and update the safetensors index."""
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    trained = mtp_head.to_safetensors_dict()
    shard_name = "model-mtp-trained.safetensors"
    save_file(trained, str(out_dir / shard_name))
    log(f"  saved {len(trained)} MTP tensors -> {out_dir / shard_name}")

    # Write a minimal index for easy inspection / diff
    index_out = {"weight_map": {k: shard_name for k in trained}}
    (out_dir / "mtp_weight_index.json").write_text(json.dumps(index_out, indent=2))


# ---------------------------------------------------------------------------
# Training progress figure
# ---------------------------------------------------------------------------

def _save_progress_figure(
    train_steps: list[int],
    train_losses: list[float],
    eval_steps: list[int],
    eval_ppls: list[float],
    eval_top1s: list[float],
    out_dir: Path,
    current_step: int,
    model_name: str = "",
    logs_name: str = "",
) -> None:
    """Save a 3-panel training progress figure to out_dir/progress_step<N>.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  matplotlib not available — skipping progress figure")
        return

    subtitle = f"{model_name}  ·  data: {logs_name}" if model_name else ""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"MTP head training  (step {current_step})\n{subtitle}", fontsize=11)

    # Panel 1: train loss
    if train_steps:
        ax1.plot(train_steps, train_losses, color="#4e79a7", linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Avg cross-entropy loss")
    ax1.set_title("Training loss")
    ax1.grid(True, alpha=0.3)

    # Panel 2: holdout PPL
    if eval_steps:
        ax2.plot(eval_steps, eval_ppls, color="#e15759", marker="o", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Perplexity (lower = better)")
    ax2.set_title("Holdout MTP PPL")
    ax2.grid(True, alpha=0.3)

    # Panel 3: top-1 accuracy
    if eval_steps:
        ax3.plot(eval_steps, eval_top1s, color="#59a14f", marker="o", linewidth=1.5)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Top-1 accuracy % (acceptance proxy)")
    ax3.set_title("Holdout top-1 accuracy")
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"progress_step{current_step:04d}.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    log(f"  saved progress figure → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", type=Path,
                   default=REPO / "out/benchmark_9b_iq3s/tesslate/model_extracted",
                   help="Path to model_extracted directory (has safetensors + config.json)")
    p.add_argument("--logs", type=Path, default=REPO / "logtrain.jsonl",
                   help="Usage-log JSONL for training data")
    p.add_argument("--out", type=Path,
                   default=REPO / "out/benchmark_9b_iq3s/tesslate/mtp_trained",
                   help="Output directory for trained MTP weights")
    p.add_argument("--steps", type=int, default=500,
                   help="Training steps")
    p.add_argument("--lr", type=float, default=2e-5,
                   help="Learning rate")
    p.add_argument("--seq-len", type=int, default=512,
                   help="Sequence length for training chunks")
    p.add_argument("--max-train-tokens", type=int, default=300_000)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--quant-type", type=str, default=None,
                   help="GGUF quant type the MTP will be deployed in (e.g. IQ3_S, Q4_K_M). "
                        "Looks up a default noise level from QUANT_NOISE_DEFAULTS. "
                        "Overridden by --quant-noise.")
    p.add_argument("--quant-noise", type=float, default=0.0,
                   help="Relative std of Gaussian noise injected into hidden states during "
                        "training to compensate for quantization error. 0 = disabled. "
                        "Use scripts/measure_quant_noise.py to calibrate for your GGUF.")
    p.add_argument("--checkpoint-every", type=int, default=100,
                   help="Save optimizer + head state every N steps for resume")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a checkpoint .pt file saved by --checkpoint-every")
    p.add_argument("--eval-every", type=int, default=100,
                   help="Evaluate MTP perplexity on holdout every N steps")
    p.add_argument("--wiki-mix", type=float, default=0.0,
                   help="Fraction of training chunks drawn from wiki/general text "
                        "(0.0 = tool-call only; 0.3 = 30%% wiki). Preserves the "
                        "head's calibration to backbone representations on general text.")
    p.add_argument("--wiki-text", type=Path, default=None,
                   help="Path to a raw .txt wiki corpus. If omitted and --wiki-mix > 0, "
                        "wikitext-2-raw-v1 is loaded from HuggingFace datasets.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iq3s-cache", type=Path, default=None,
                   help="Path to a hidden-state cache .npy produced by "
                        "scripts/capture_iq3s_hidden_states.py. When set, the backbone "
                        "forward pass is skipped and h is read from the cache instead. "
                        "Forces --quant-noise to 0 (real IQ3_S noise is baked into the cache).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log(f"device: {device}")

    # --- Config ---
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(args.model))
    # Derive a human-readable model name for figure titles
    model_name: str = (
        getattr(config, "_name_or_path", None)
        or getattr(config, "name_or_path", None)
        or args.model.name   # directory name as fallback (e.g. "model_extracted")
    )
    # If the config name is just a local path fragment, use the parent dir name
    if "/" not in model_name and model_name in ("model_extracted", "."):
        model_name = args.model.parent.name
    logs_name: str = args.logs.name
    log(f"model:       {model_name}  ({args.model})")
    log(f"arch:        {config.architectures}")
    log(f"hidden_size: {config.hidden_size}")
    log(f"steps:       {args.steps}  lr={args.lr}  seq_len={args.seq_len}")
    log(f"out:         {args.out}")

    if args.iq3s_cache is not None:
        import numpy as np
        if not args.iq3s_cache.exists():
            log(f"ERROR: --iq3s-cache not found: {args.iq3s_cache}")
            return 1
        _ids_path = args.iq3s_cache.with_name(
            args.iq3s_cache.stem + "_chunk_ids.npy")
        if not _ids_path.exists():
            log(f"ERROR: companion chunk_ids file not found: {_ids_path}")
            return 1
        _peek = np.load(str(args.iq3s_cache), mmap_mode="r")
        log(f"iq3s cache:  {args.iq3s_cache}  shape={_peek.shape}  dtype={_peek.dtype}")
        del _peek

    if args.dry_run:
        log("dry-run — exiting")
        return 0

    # --- Load backbone (frozen) ---
    with phase("load backbone (frozen)"):
        # Patch arch to load as CausalLM (text weights are identical)
        config_cpy = copy.deepcopy(config)
        config_cpy.architectures = ["Qwen3_5ForCausalLM"]
        config_cpy.layer_types   = getattr(config, "layer_types", None) or \
                                   ["full_attention"] * config.num_hidden_layers

        from transformers import Qwen3_5ForCausalLM
        backbone = Qwen3_5ForCausalLM.from_pretrained(
            str(args.model),
            config=config_cpy,
            dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
        ).to(device).eval()
        for param in backbone.parameters():
            param.requires_grad_(False)
        log(f"  backbone loaded, {sum(p.numel() for p in backbone.parameters()):,} params (frozen)")

    # --- Build and initialise MTP head ---
    with phase("build MTP head"):
        mtp_head = Qwen3MTPHead(config_cpy).to(device).to(torch.bfloat16)
        mtp_tensors = _load_mtp_tensors(args.model)
        mtp_head.load_from_safetensors(mtp_tensors)
        n_mtp = sum(p.numel() for p in mtp_head.parameters())
        log(f"  MTP head: {n_mtp:,} trainable params")

    # --- Data ---
    with phase("tokenise training data"):
        chunks = build_token_batches(
            args.model, args.logs,
            seq_len=args.seq_len,
            max_tokens=args.max_train_tokens,
            seed=args.seed,
            wiki_mix=args.wiki_mix,
            wiki_text=args.wiki_text,
        )
        log(f"  {len(chunks)} chunks of length {args.seq_len}")

    if not chunks:
        log("ERROR: no training data — check --logs path")
        return 1

    # --- IQ3_S cache (optional) ---
    iq3s_cache = None
    iq3s_chunk_ids = None
    if args.iq3s_cache is not None:
        import numpy as np
        if not args.iq3s_cache.exists():
            log(f"ERROR: --iq3s-cache not found: {args.iq3s_cache}")
            return 1
        ids_path = args.iq3s_cache.with_name(
            args.iq3s_cache.stem + "_chunk_ids.npy")
        if not ids_path.exists():
            log(f"ERROR: companion chunk_ids file not found: {ids_path}")
            return 1
        iq3s_cache = np.load(str(args.iq3s_cache), mmap_mode="r")
        iq3s_chunk_ids = np.load(str(ids_path), mmap_mode="r")

        if iq3s_cache.shape[1] != args.seq_len:
            log(f"ERROR: cache seq_len {iq3s_cache.shape[1]} != --seq-len {args.seq_len}")
            return 1
        if iq3s_cache.shape[2] != config.hidden_size:
            log(f"ERROR: cache hidden {iq3s_cache.shape[2]} != config.hidden_size {config.hidden_size}")
            return 1
        if iq3s_chunk_ids.shape[0] != iq3s_cache.shape[0]:
            log(f"ERROR: chunk_ids rows {iq3s_chunk_ids.shape[0]} != cache rows {iq3s_cache.shape[0]}")
            return 1
        if iq3s_chunk_ids.shape[1] != args.seq_len + 2:
            log(f"ERROR: chunk_ids width {iq3s_chunk_ids.shape[1]} != seq_len+2 {args.seq_len + 2}")
            return 1
        if iq3s_chunk_ids.shape[0] > len(chunks):
            log(f"ERROR: cache n_chunks {iq3s_chunk_ids.shape[0]} > tokenised chunks {len(chunks)} "
                "(seed / max_tokens / wiki_mix drift between capture and training)")
            return 1
        # Token-level parity check on the first chunk.
        if list(iq3s_chunk_ids[0].astype(int)) != list(chunks[0]):
            log("ERROR: cached chunk_ids[0] != tokenised chunks[0] — capture used different inputs")
            log(f"  cached[:10]:    {list(iq3s_chunk_ids[0][:10].astype(int))}")
            log(f"  tokenised[:10]: {chunks[0][:10]}")
            return 1
        if iq3s_chunk_ids.shape[0] < len(chunks):
            log(f"  cache holds {iq3s_chunk_ids.shape[0]} chunks (< {len(chunks)} tokenised); "
                "training will cycle through those only")
            chunks = chunks[: iq3s_chunk_ids.shape[0]]
        log(f"IQ3_S cache mode active: {args.iq3s_cache}")
        log(f"  cache shape: {iq3s_cache.shape} dtype={iq3s_cache.dtype}")

    # --- Optimiser ---
    opt = torch.optim.AdamW(mtp_head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # --- Resume from checkpoint ---
    step = 0
    chunk_idx = 0
    train_steps: list[int] = []
    train_losses: list[float] = []
    eval_steps: list[int] = []
    eval_ppls: list[float] = []
    eval_top1s: list[float] = []

    if args.resume is not None:
        if not args.resume.exists():
            log(f"ERROR: checkpoint not found: {args.resume}")
            return 1
        ckpt = torch.load(str(args.resume), map_location=device)
        mtp_head.load_state_dict(ckpt["mtp_head"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        chunk_idx = ckpt.get("chunk_idx", step * args.grad_accum)
        train_steps = ckpt.get("train_steps", [])
        train_losses = ckpt.get("train_losses", [])
        eval_steps = ckpt.get("eval_steps", [])
        eval_ppls = ckpt.get("eval_ppls", [])
        eval_top1s = ckpt.get("eval_top1s", [])
        log(f"  resumed from step {step} ({args.resume.name})")

    # --- Resolve quantization noise level ---
    quant_noise = args.quant_noise
    if iq3s_cache is not None:
        if quant_noise != 0.0 or args.quant_type is not None:
            log("IQ3_S cache mode: forcing quant_noise=0 "
                "(real IQ3_S noise is baked into cached hidden states)")
        quant_noise = 0.0
    elif quant_noise == 0.0 and args.quant_type is not None:
        qt = args.quant_type.upper().replace("-", "_")
        quant_noise = QUANT_NOISE_DEFAULTS.get(qt, 0.0)
        if quant_noise > 0:
            log(f"quant noise: {quant_noise:.4f} (from defaults table for {qt})")
        else:
            log(f"WARNING: quant type '{qt}' not in defaults table — noise injection disabled")
    elif quant_noise > 0:
        log(f"quant noise: {quant_noise:.4f} (from --quant-noise)")
    else:
        log("quant noise: disabled (pass --quant-type or --quant-noise to enable)")

    # --- Training loop ---
    log(f"training for {args.steps} steps (grad_accum={args.grad_accum}) …")
    mtp_head.train()
    accum_loss = 0.0
    accum_tokens = 0
    t0 = time.time()
    opt.zero_grad()

    while step < args.steps:
        idx = chunk_idx % len(chunks)
        chunk = chunks[idx]
        chunk_idx += 1

        if iq3s_cache is not None:
            row = iq3s_chunk_ids[idx]
            input_ids = torch.from_numpy(row[:-2].astype("int64")).to(device).unsqueeze(0)
            next_ids  = torch.from_numpy(row[1:-1].astype("int64")).to(device).unsqueeze(0)
            labels    = torch.from_numpy(row[2:].astype("int64")).to(device).unsqueeze(0)
        else:
            input_ids = torch.tensor([chunk[:-2]], dtype=torch.long, device=device)
            next_ids  = torch.tensor([chunk[1:-1]], dtype=torch.long, device=device)
            labels    = torch.tensor([chunk[2:]], dtype=torch.long, device=device)

        with torch.no_grad():
            if iq3s_cache is not None:
                h = torch.from_numpy(iq3s_cache[idx].copy()).to(
                    device, dtype=torch.bfloat16).unsqueeze(0)  # [1, T, H]
            else:
                out = backbone(input_ids=input_ids, output_hidden_states=True)
                h   = out.hidden_states[-1]                     # [1, T, H]
            e   = backbone.model.embed_tokens(next_ids)         # [1, T, H]

        # Inject quantization noise: simulate the activation shift the MTP head
        # will see at inference when the backbone runs at reduced precision.
        # Noise is proportional to the local activation norm so it scales
        # correctly across layers and sequence positions.
        if quant_noise > 0.0:
            h_scale = h.detach().norm(dim=-1, keepdim=True)
            h = h + quant_noise * h_scale * torch.randn_like(h)
            # Embeddings pass through far fewer quantized layers (just the one
            # embedding table), so noise is ~1/sqrt(n_layers) of h's noise.
            e_noise = quant_noise / math.sqrt(
                backbone.config.num_hidden_layers)
            e_scale = e.detach().norm(dim=-1, keepdim=True)
            e = e + e_noise * e_scale * torch.randn_like(e)

        B, T, _ = h.shape
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        with torch.no_grad():
            rotary = backbone.model.rotary_emb(h, pos_ids)

        mtp_out = mtp_head(h, e, rotary, position_ids=pos_ids)
        logits  = backbone.lm_head(mtp_out)                 # [1, T, V]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        ) / args.grad_accum
        loss.backward()

        accum_loss   += loss.item() * args.grad_accum
        accum_tokens += labels.numel()

        if (chunk_idx % args.grad_accum) == 0:
            torch.nn.utils.clip_grad_norm_(mtp_head.parameters(), 1.0)
            opt.step()
            scheduler.step()
            opt.zero_grad()
            step += 1

            if step % 10 == 0 or step == 1:
                elapsed = time.time() - t0
                avg_loss = accum_loss / (step * args.grad_accum) if step else accum_loss
                log(f"  step {step:4d}/{args.steps}  loss={avg_loss:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}  "
                    f"elapsed={elapsed/60:.1f}m")
                train_steps.append(step)
                train_losses.append(avg_loss)

            if step == 25 or step % args.eval_every == 0:
                mtp_head.eval()
                with phase(f"eval MTP PPL at step {step}"):
                    ppl, top1 = _eval_mtp_ppl(
                        backbone, mtp_head, device,
                        args.model, args.logs,
                        seq_len=args.seq_len,
                    )
                    log(f"  holdout MTP PPL = {ppl:.2f}  top-1 acc = {top1*100:.2f}%  (acceptance rate proxy)")
                    eval_steps.append(step)
                    eval_ppls.append(ppl)
                    eval_top1s.append(top1 * 100)
                    _save_progress_figure(
                        train_steps, train_losses,
                        eval_steps, eval_ppls, eval_top1s,
                        args.out, step,
                        model_name=model_name,
                        logs_name=logs_name,
                    )

                if step % args.checkpoint_every == 0:
                    ckpt_path = args.out / f"checkpoint_step{step:04d}.pt"
                    args.out.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "step": step,
                        "chunk_idx": chunk_idx,
                        "mtp_head": mtp_head.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "train_steps": train_steps,
                        "train_losses": train_losses,
                        "eval_steps": eval_steps,
                        "eval_ppls": eval_ppls,
                        "eval_top1s": eval_top1s,
                    }, str(ckpt_path))
                    log(f"  saved checkpoint → {ckpt_path.name}")

                mtp_head.train()

    # Final eval
    mtp_head.eval()
    with phase("final holdout eval"):
        ppl, top1 = _eval_mtp_ppl(
            backbone, mtp_head, device,
            args.model, args.logs,
            seq_len=args.seq_len,
        )
        log(f"  final holdout MTP PPL = {ppl:.2f}  top-1 acc = {top1*100:.2f}%")
        eval_steps.append(args.steps)
        eval_ppls.append(ppl)
        eval_top1s.append(top1 * 100)
        _save_progress_figure(
            train_steps, train_losses,
            eval_steps, eval_ppls, eval_top1s,
            args.out, args.steps,
        )

    # --- Save ---
    with phase("save trained MTP weights"):
        _save_mtp_weights(mtp_head, args.model, args.out)

    elapsed = (time.time() - t0) / 60
    log(f"=== DONE in {elapsed:.1f} min ===")
    log("Next steps:")
    log("  1. Install trained weights into model_extracted:")
    log(f"       cp {args.out}/model-mtp-trained.safetensors \\")
    log(f"          {args.model}/model-mtp-trained.safetensors")
    log("     # update model.safetensors.index.json to point mtp.* to model-mtp-trained.safetensors")
    log("  2. Delete old F16 GGUF and reconvert:")
    log(f"       rm {args.model.parents[0]}/model-f16.gguf")
    log("       uv run quant-tuner run ... (or run the convert step directly)")
    log("  3. Evaluate acceptance rate:")
    log("       uv run python scripts/bench_mtp_speed.py \\")
    log("           --model <new-gguf> --holdout-jsonl logtrain.jsonl \\")
    log("           --reps 5 --n-max 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
