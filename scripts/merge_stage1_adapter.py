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
    "Search the repo for every call site of `run_ptq` and report which files they are in.",
    "Read config.yaml, change the batch size to 8, and write it back.",
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
        enc = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, return_tensors="pt")
        # transformers 5.x returns a BatchEncoding here, not a bare tensor.
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        with torch.no_grad():
            out.append(model(ids.to(device)).logits[0, -1].float().cpu())
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
    ap.add_argument("--max-kl", type=float, default=1e-2,
                    help="Max allowed KL(base+adapter || merged) in nats, per prompt")
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
        d_base = (merged_logits - base_logits).abs().max().item()

        # Compare DISTRIBUTIONS, not raw logits. An absolute logit tolerance is
        # the wrong test and initially failed a correct merge: the unmerged path
        # computes Wx + B(Ax) while the merged path computes (W+BA)x, and in
        # bf16 (~8 mantissa bits) that reassociation accumulates over 42 layers
        # to a max deviation of ~0.7 on logits whose scale is +/-29. Measured on
        # this model the two agree to KL ~1e-4 nats with identical top-1 and
        # top-5 -- while 4-bit quantization, which we consider acceptable, sits
        # nearer 1e-2. What matters is whether the model behaves the same, and
        # KL plus top-1 agreement says that directly; a logit atol only says
        # how big the logits happened to be.
        kls, top1 = [], []
        for i in range(len(PROMPTS)):
            pr = torch.softmax(ref_logits[i], -1)
            pm = torch.softmax(merged_logits[i], -1)
            kls.append((pr * (pr.clamp_min(1e-12).log()
                              - pm.clamp_min(1e-12).log())).sum().item())
            top1.append(bool(ref_logits[i].argmax() == merged_logits[i].argmax()))
        worst = max(kls)
        print(f"[verify] KL(base+adapter || merged)  max {worst:.3e} over "
              f"{len(PROMPTS)} prompts   (must be < {a.max_kl:.0e})")
        print(f"[verify] top-1 agreement             {sum(top1)}/{len(top1)}")
        print(f"[verify] |merged - base| max logit   {d_base:.4f}   (must be > 0.001)")
        if worst > a.max_kl:
            raise SystemExit(
                f"Merged model's output distribution differs from base+adapter "
                f"(KL {worst:.3e} > {a.max_kl:.0e}). Do not publish.")
        if not all(top1):
            raise SystemExit(
                "Merged model picks a different top-1 token than base+adapter "
                "on at least one prompt. Do not publish.")
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
