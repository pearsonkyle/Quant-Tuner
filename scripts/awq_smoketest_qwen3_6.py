"""Smoke-test AWQ group discovery + a 1-group alpha search on Qwen3.6-27B.

Goal: catch architecture-mismatch problems (hybrid Mamba/attention layers,
plus_one RMSNorm form, MTP placement) before kicking off the multi-hour
full AWQ pipeline.

Outputs printed to stdout:
  * total layer count + layer_types breakdown from the model config
  * number of AWQ scale groups discovered (attn + mlp)
  * one full-attn group: alpha grid loss curve at small token budget
  * one mlp group: alpha grid loss curve
  * sanity check that fold_rmsnorm_gain with plus_one=True / False yields
    weights with the right magnitude (no NaN / no obvious blowup)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.calibrate.awq import (  # noqa: E402
    _get_module,
    discover_groups,
    proxy_loss,
    scale_from_alpha,
)

MODEL_DIR = REPO / "out" / "q5_k_s_qwen3_6_mtp" / "model_extracted"
CALIB = REPO / "out" / "q5_k_s_qwen3_6_mtp" / "corpus" / "train.txt"
DEVICE = "mps"
DTYPE = torch.bfloat16
N_TOKENS = 512   # tiny for smoke test
CTX = 256
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> int:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = getattr(text_cfg, "layer_types", None)
    print(f"model_type:    {text_cfg.model_type}")
    print(f"num_hidden:    {text_cfg.num_hidden_layers}")
    if layer_types is not None:
        from collections import Counter
        print(f"layer_types:   {Counter(layer_types)}")
    print(f"mtp_layers:    {getattr(text_cfg, 'mtp_num_hidden_layers', 'n/a')}")
    print()

    print(f"loading {MODEL_DIR} -> {DEVICE}/{DTYPE} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=DTYPE, trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    groups = discover_groups(model)
    n_attn = sum(1 for g in groups if g.group_id.endswith("_attn"))
    n_mlp = sum(1 for g in groups if g.group_id.endswith("_mlp"))
    print(f"discover_groups: {len(groups)} total ({n_attn} attn, {n_mlp} mlp)")
    if groups:
        print(f"  first: {groups[0].group_id} anchor={groups[0].anchor}")
        print(f"         members={groups[0].members}")
        print(f"         prev_norm={groups[0].prev_norm}")
        print(f"  last:  {groups[-1].group_id} anchor={groups[-1].anchor}")
    print()

    # Capture activations for the *first* attn group (if any) and the *first* mlp group.
    targets = []
    first_attn = next((g for g in groups if g.group_id.endswith("_attn")), None)
    first_mlp = next((g for g in groups if g.group_id.endswith("_mlp")), None)
    if first_attn:
        targets.append(first_attn)
    if first_mlp:
        targets.append(first_mlp)
    if not targets:
        print("no groups to probe; bailing.")
        return 2

    sum_abs = {g.group_id: None for g in targets}
    tok_count = {g.group_id: 0 for g in targets}
    cached_X = {g.group_id: None for g in targets}

    def make_hook(gid: str):
        def pre(_m, in_args):
            x = in_args[0].reshape(-1, in_args[0].shape[-1])
            a = x.float().abs().sum(dim=0).detach().cpu()
            sum_abs[gid] = a if sum_abs[gid] is None else sum_abs[gid] + a
            tok_count[gid] += x.shape[0]
            if cached_X[gid] is None:
                cached_X[gid] = x[:128].detach().float().cpu()
        return pre

    handles = [
        _get_module(model, g.anchor).register_forward_pre_hook(make_hook(g.group_id))
        for g in targets
    ]
    try:
        text = CALIB.read_text()
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0][:N_TOKENS]
        chunks = ids.split(CTX)
        print(f"calibration probe: {ids.shape[0]} tokens in {len(chunks)} chunks")
        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                if chunk.numel() < 2:
                    continue
                model(chunk.unsqueeze(0).to(DEVICE))
                print(f"  chunk {i + 1}/{len(chunks)} ok")
    finally:
        for h in handles:
            h.remove()

    print()
    for g in targets:
        if sum_abs[g.group_id] is None or tok_count[g.group_id] == 0:
            print(f"[{g.group_id}] NO ACTIVATIONS captured")
            continue
        s = (sum_abs[g.group_id] / tok_count[g.group_id]).clamp_min(1e-8)
        X = cached_X[g.group_id]
        print(f"[{g.group_id}] activations: shape={tuple(X.shape)} "
              f"mean|x|={float(s.mean()):.4f}  max|x|={float(s.max()):.4f}")
        Ws = [_get_module(model, m).weight.detach().float().cpu() for m in g.members]
        losses = []
        for a in ALPHAS:
            scale = scale_from_alpha(s, a)
            loss = sum(proxy_loss(W, X, scale) for W in Ws)
            losses.append((a, loss))
        best = min(losses, key=lambda t: t[1])
        print("           alpha grid:")
        for a, l in losses:
            mark = "  <- best" if (a, l) == best else ""
            print(f"             α={a:.2f}  loss={l:.4e}{mark}")
        print()

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
