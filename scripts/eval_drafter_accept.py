"""Measure drafter next-token argmax match (== speculative acceptance proxy) on
held-out windows, for one or more drafter checkpoints against a shared target.

Under greedy speculative decoding, a drafted token is accepted iff it equals the
target's argmax. So position-wise argmax-match of the drafter (given target
context) is a direct proxy for acceptance rate — the metric that governs speedup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Gemma4AssistantForCausalLM,
)


def match_rate(target, assistant, ids, device, chunk=1024) -> tuple[int, int]:
    matched = total = 0
    for start in range(0, len(ids), chunk):
        piece = ids[start : start + chunk]
        if len(piece) < 2:
            continue
        input_ids = torch.tensor([piece], device=device)
        with torch.no_grad():
            tgt = target.model(
                input_ids=input_ids, return_shared_kv_states=True,
                output_hidden_states=True, use_cache=False,
            )
            emb = torch.cat(
                [target.get_input_embeddings()(input_ids), tgt.hidden_states[-1]], dim=-1
            )
            logits = assistant(
                inputs_embeds=emb, shared_kv_states=tgt.shared_kv_states,
                position_ids=torch.arange(input_ids.shape[1], device=device)[None],
            ).logits
            pred = logits[:, :-1].argmax(-1)
            matched += (pred == input_ids[:, 1:]).sum().item()
            total += input_ids.shape[1] - 1
    return matched, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafters", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--windows", required=True, type=Path)
    ap.add_argument("--max-windows", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, device_map={"": args.device},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        ),
    ).eval()

    windows = [json.loads(line)["input_ids"] for line in open(args.windows)][: args.max_windows]

    for pair in args.drafters:
        name, path = pair.split("=", 1)
        asst = Gemma4AssistantForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(args.device).eval()
        m = t = 0
        for ids in windows:
            wm, wt = match_rate(target, asst, ids, args.device)
            m += wm; t += wt
        print(f"{name}: acceptance-proxy (argmax match) = {100 * m / t:.2f}%  ({m}/{t} tokens)")
        del asst
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
