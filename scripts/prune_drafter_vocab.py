"""Gather a Gemma-4 MTP assistant (drafter) down to a keepset vocabulary.

Speculative decoding requires draft and target to share a token id space, so a
target pruned to 65,536 needs its drafter pruned to the SAME keepset.
`prune_model_vocab.py` in LLM-Training-Kit cannot do it: the assistant is
`Gemma4AssistantForCausalLM`, whose head is not a plain tied embedding.

The drafter is 78.8M params of which 67.4M (85.5%) is `model.embed_tokens`
(262144 x 256), tied to `lm_head`. Cutting to 65,536 takes it to ~28M params
(159 MB -> ~57 MB) and shrinks the logits tensor 4x.

## Why this drops the ordered-embedding head

The stock drafter sets `use_ordered_embeddings: true`, routing the head through
`Gemma4AssistantMaskedEmbedder`: `token_ordering` is a BALANCED PERMUTATION
reshaped to `(num_centroids, vocab // num_centroids)`, and the forward scores
only the tokens owned by the top-k centroids. Note what it does NOT do -- it
still allocates the full `(batch, seq, vocab_size)` output and scatters into it.
The saving is matmul, not memory.

That saving is worth having at 262k and largely stops being worth it at 65k:

    backbone (4 layers, hidden 256)        7.3M MACs/token
    masked head  @262k  1.57M (0.21x)   |  dense head @262k  67.1M (9.14x)
    masked head   @65k  0.79M (0.11x)   |  dense head  @65k  16.8M (2.29x)

Against a ~5B-param target, seven drafts per verify cost 57M MACs masked versus
169M dense -- both a couple of percent of one target forward. The trick was
built for the 9x case.

Keeping it would also mean rebuilding the partition, and that cannot be
validated offline. Measured on this keepset: the kept tokens fall wildly
unevenly across the stock centroids (435 own zero, 23 own all 128, 8 own exactly
32; 52% of tokens would have to move), so membership must be re-derived rather
than repaired. But the stock partition has no recoverable relationship to
embedding geometry to re-derive it FROM -- scoring tokens against centroid
directions puts a token's own centroid in the top-32 for 0.14% of tokens, below
the 1.56% you would get by chance, and cos(centroid_c, mean embedding of cluster
c) is 0.473 against a 0.422 shuffled control. `token_ordering` is an `nn.Buffer`,
never trained, so a partition guessed wrong stays wrong.

Dense is exact, has no partition, and costs ~2% of speculative-decode compute.
If a benchmark later shows drafter latency actually matters, the ordered path
can come back -- but it needs real hidden-state statistics from a coupled
target+drafter forward to place tokens, not embedding geometry.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--keepset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", choices=("dense", "ordered"), default="dense",
                    help="dense drops the masked-embedding head (see module docstring)")
    args = ap.parse_args()

    if args.head == "ordered":
        raise SystemExit(
            "ordered head not implemented: rebuilding token_ordering needs hidden-state "
            "statistics from a coupled target+drafter forward. Embedding geometry does "
            "not recover the stock partition (measured: 0.14% top-32 self-recall against "
            "1.56% chance). See the module docstring."
        )

    src, out = Path(args.drafter), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keep = json.load(open(args.keepset))
    assert keep == sorted(keep), "keepset must be ascending (new id j == keepset[j])"
    keep_t = torch.tensor(keep, dtype=torch.long)
    new_vocab = len(keep)

    cfg = json.loads((src / "config.json").read_text())
    tcfg = cfg.get("text_config", cfg)
    old_vocab = tcfg["vocab_size"]

    weights = load_file(str(src / "model.safetensors"))
    before = sum(v.numel() for v in weights.values())

    gathered, dropped = [], []
    for k in list(weights):
        if k.startswith("masked_embedding."):
            del weights[k]
            dropped.append(k)
            continue
        v = weights[k]
        if v.ndim >= 1 and v.shape[0] == old_vocab:
            weights[k] = v.index_select(0, keep_t).contiguous()
            gathered.append(k)

    if "model.embed_tokens.weight" not in gathered:
        raise SystemExit("embed_tokens was not gathered -- vocab axis not where expected")

    cfg["use_ordered_embeddings"] = False
    for c in (cfg, tcfg):
        if "vocab_size" in c:
            c["vocab_size"] = new_vocab
    # Leave num_centroids / centroid_intermediate_top_k in place. They are inert
    # while use_ordered_embeddings is false, and removing them would let a
    # transformers build that constructs the masked embedder unconditionally fall
    # back to ITS defaults for the shape instead of erroring.

    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    save_file(weights, str(out / "model.safetensors"), metadata={"format": "pt"})
    for extra in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        if (src / extra).exists():
            shutil.copy2(src / extra, out / extra)

    after = sum(v.numel() for v in weights.values())
    report = {
        "old_vocab": old_vocab,
        "new_vocab": new_vocab,
        "head": "dense (use_ordered_embeddings=false)",
        "tensors_gathered": gathered,
        "tensors_dropped": dropped,
        "params": {"before": before, "after": after, "shrink": round(before / after, 3)},
        "bytes_bf16": {"before": before * 2, "after": after * 2},
    }
    (out / "prune_drafter_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
