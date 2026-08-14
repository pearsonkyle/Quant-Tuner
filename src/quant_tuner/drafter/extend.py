"""Grow a Gemma-4 MTP assistant (drafter) by one decoder layer.

The stock drafter is 4 KV-shared layers ``[sliding, sliding, sliding, full]``.
More layers = more capacity to model the target's next token → potentially higher
speculative acceptance, at the cost of a slightly slower draft. The new layer is
randomly-initialized capacity that needs real training data (this is why the
FinePhrase scale matters) — but we warm-init it from an existing sliding layer so
the residual stream isn't shocked at step 0.

Insertion keeps the ``full`` layer last (its role is unchanged): the new sliding
layer goes at the end of the sliding run, so
``[sliding, sliding, sliding, full]`` → ``[sliding, sliding, sliding, sliding, full]``.
All layers stay KV-shared (``num_kv_shared_layers == num_hidden_layers``, which the
config asserts).

Torch/safetensors import at call time so the module stays importable for tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def extend_drafter(src: str | Path, dst: str | Path, warm_from: int = 2,
                   identity_init: bool = True) -> Path:
    """Write a +1-layer assistant to ``dst``. The new sliding layer's weights are
    copied from decoder layer ``warm_from``, then (``identity_init``, default) its
    residual-writing outputs — ``self_attn.o_proj`` and ``mlp.down_proj`` — are
    ZEROED so the layer is an identity residual at init: the 5-layer model starts
    numerically identical to the 4-layer one (same acceptance) and training can
    only add. Without this, inserting a non-identity layer wrecks the tuned
    residual stream (observed: acceptance 55% -> 24%) and the hard-to-train head
    can't recover it. Gradients still flow to o_proj/down_proj (their inputs are
    non-zero), so they learn from zero. Returns ``dst``."""
    import torch  # noqa: F401
    from safetensors.torch import load_file, save_file

    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    n = tc["num_hidden_layers"]
    types = list(tc["layer_types"])
    if types[-1] != "full_attention":
        raise ValueError(f"expected full_attention last, got {types}")

    # new sliding layer at index n-1 (end of the sliding run); full shifts to n.
    new_idx = n - 1
    if types[warm_from] != "sliding_attention":
        raise ValueError(f"warm_from layer {warm_from} must be sliding, got {types[warm_from]}")

    weights = load_file(str(src / "model.safetensors"))
    out: dict = {}
    for k, v in weights.items():
        m = re.search(r"(.*layers\.)(\d+)(\..*)", k)
        if not m:
            out[k] = v.clone()
            continue
        li = int(m.group(2))
        # layers below the insertion point keep their index; the old `full`
        # layer (last) shifts up by one.
        new_li = li if li < new_idx else li + 1
        out[f"{m.group(1)}{new_li}{m.group(3)}"] = v.clone()

    # materialize the new layer from `warm_from`
    for k, v in weights.items():
        m = re.search(rf"(.*layers\.){warm_from}(\..*)", k)
        if m:
            nk = f"{m.group(1)}{new_idx}{m.group(2)}"
            if identity_init and (nk.endswith("self_attn.o_proj.weight")
                                  or nk.endswith("mlp.down_proj.weight")):
                out[nk] = torch.zeros_like(v)  # zero the attn + mlp branches
            elif identity_init and nk.endswith("layer_scalar"):
                # layer_scalar multiplies the WHOLE layer output (residual + branch);
                # must be 1 for identity, else the residual x itself gets rescaled.
                out[nk] = torch.ones_like(v)
            else:
                out[nk] = v.clone()

    tc["num_hidden_layers"] = n + 1
    tc["num_kv_shared_layers"] = n + 1
    tc["layer_types"] = types[:new_idx] + ["sliding_attention"] + types[new_idx:]

    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for extra in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        p = src / extra
        if p.exists():
            (dst / extra).write_text(p.read_text())
    return dst
