#!/usr/bin/env python3
"""Merge the stage-1 LoRA into the stage-0 base and write a plain bf16 model.

This is the hinge of the release: the merged directory is BOTH the artifact
published as the main model AND the only thing either quantizer can consume.
Neither GPTQ nor llama.cpp's converter understands a PEFT adapter, so a wrong
merge does not fail loudly -- it produces a model that loads, generates
fluent text, and has quietly lost the training.

So the merge is verified rather than assumed. `--verify` runs the same prompts
through PeftModel(base, adapter) and through the merged model and compares
logits. That is the only check that actually distinguishes "merged" from
"loaded the base and saved it again", which is the failure this guards:

  - adapter_config.json here carries a `base_model_name_or_path` pointing at a
    path from another machine (/home/pearsonkyle/...). Passing an already-loaded
    model to PeftModel.from_pretrained bypasses that lookup; letting peft
    resolve it would silently fetch or fail.
  - a merge that matched the base within float noise means the adapter
    contributed nothing, whatever the file sizes say.

Context: the stage-0 directory is named ...-32k-... for the context it was
TRAINED at; its config already declares max_position_embeddings=131072, which
is what stage 1 trained against and what the merged model must keep.

    python scripts/merge_stage1_adapter.py \
        --base /workspace/models/gemma4-e4b-stage0-32k-v65536/final \
        --adapter /workspace/models/gemma4-e4b-stage1-131k-v65536/checkpoint-7000 \
        --out /workspace/models/gemma4-e4b-coder-merged \
        --verify
"""
import argparse
import json
import shutil
from pathlib import Path

import torch


PROMPTS = [
    "Write a Python function that reads a JSONL file and yields parsed rows.",
    "List the files in the current directory, then summarise what the project does.",
    "What is the capital of France?",
]


def load_base(path: str, device: str):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa")


def logits_for(model, tok, device):
    """Last-token logits for each probe prompt, as one stacked tensor."""
    out = []
    model.eval()
    for p in PROMPTS:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out.append(model(ids).logits[0, -1].float().cpu())
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify", action="store_true",
                    help="Compare merged logits against base+adapter, and against "
                         "the bare base. Costs one short forward pass per prompt.")
    ap.add_argument("--atol", type=float, default=0.05,
                    help="Max allowed |merged - (base+adapter)| on last-token logits")
    a = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer

    out = Path(a.out)
    tok = AutoTokenizer.from_pretrained(a.base)

    base_logits = ref_logits = None
    if a.verify:
        print("[verify] scoring the bare base")
        m = load_base(a.base, a.device)
        base_logits = logits_for(m, tok, a.device)
        print("[verify] scoring base + adapter (unmerged)")
        m = PeftModel.from_pretrained(m, a.adapter)
        ref_logits = logits_for(m, tok, a.device)
        del m
        torch.cuda.empty_cache()

        drift = (ref_logits - base_logits).abs().max().item()
        print(f"[verify] adapter changes the base by max |dlogit| = {drift:.4f}")
        if drift < 1e-3:
            raise SystemExit(
                "The adapter barely moves the base's logits. Either the wrong "
                "checkpoint was passed or the adapter failed to load -- merging "
                "this would publish the base model under the trained model's name.")

    print(f"loading base  {a.base}")
    model = load_base(a.base, a.device)
    print(f"loading adapter {a.adapter}")
    model = PeftModel.from_pretrained(model, a.adapter)
    print("merging")
    model = model.merge_and_unload()

    if a.verify:
        merged_logits = logits_for(model, tok, a.device)
        d_ref = (merged_logits - ref_logits).abs().max().item()
        d_base = (merged_logits - base_logits).abs().max().item()
        print(f"[verify] |merged - (base+adapter)| = {d_ref:.4f}   (must be < {a.atol})")
        print(f"[verify] |merged - base|           = {d_base:.4f}   (must be > 0.001)")
        if d_ref > a.atol:
            raise SystemExit("Merged model does not reproduce base+adapter. Do not publish.")
        if d_base < 1e-3:
            raise SystemExit("Merged model is indistinguishable from the base. Do not publish.")
        print("[verify] OK -- merge reproduces the adapter and differs from the base")

    print(f"saving {out}")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    # save_pretrained does not carry these, and a model without its chat
    # template is unusable for the task it was trained on.
    for name in ("chat_template.jinja", "generation_config.json"):
        src = Path(a.base) / name
        if src.exists():
            shutil.copy2(src, out / name)
            print(f"  copied {name}")
        else:
            print(f"  WARNING: {name} not found in base")

    cfg = json.loads((out / "config.json").read_text())
    print(f"\nvocab_size              {cfg.get('vocab_size')}")
    print(f"max_position_embeddings {cfg.get('max_position_embeddings')}")
    print(f"dtype                   {cfg.get('dtype') or cfg.get('torch_dtype')}")
    if not (out / "chat_template.jinja").exists():
        print("\nWARNING: no chat template in the output; tool-call formatting will be wrong.")
    print("\nMERGE DONE")


if __name__ == "__main__":
    raise SystemExit(main())
