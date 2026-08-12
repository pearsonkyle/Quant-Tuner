#!/usr/bin/env python3
"""Tool-call-preserving QLoRA SFT of gemma-4-E4B.

Fixes the tool-calling regression from the flattened-data SFT. Two changes:
  1. Use the BASE gemma-4 chat template (renders <|tool_call>call:...<tool_call|>
     and passes `tools`), NOT unsloth's get_chat_template override which drops tools.
  2. Custom label masking: the template strings all tool_calls + tool_responses
     under ONE <|turn>model, so response-only masking would train the model on
     tool RESULTS (teaching it to hallucinate outputs). We instead unmask only the
     assistant's generated spans (its tool_calls + text) and mask the
     <|tool_response> environment spans + user/system.

Data: pearsonkyle/swe-agentic-trajectories (structured tool-use, upsampled) +
pearsonkyle/broad-domain-supplement instruct (breadth, no tools).
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

SWE = "https://huggingface.co/datasets/pearsonkyle/swe-agentic-trajectories/resolve/main/data/resolved.jsonl"
BROAD = "https://huggingface.co/datasets/pearsonkyle/broad-domain-supplement/resolve/main/data/instruct.jsonl"

_MARKERS = re.compile(
    r"<\|turn>model\n|<\|turn>user|<\|turn>system|<\|tool_call>|<tool_call\|>|<\|tool_response>"
)


def unmask_spans(text: str) -> list[tuple[int, int]]:
    """Char spans of assistant-generated tokens (tool_calls + model text)."""
    spans, assistant, start = [], False, 0
    for m in _MARKERS.finditer(text):
        mk = m.group()
        if mk == "<|turn>model\n":
            assistant, start = True, m.end()
        elif mk in ("<|turn>user", "<|turn>system") or mk == "<|tool_response>":
            if assistant:
                spans.append((start, m.start()))
                assistant = False
        elif mk == "<|tool_call>":
            if not assistant:
                assistant, start = True, m.start()
    if assistant:
        spans.append((start, len(text)))
    return spans


def _iter(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        for line in r:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_examples(tokenizer, max_len, n_swe, swe_repeat, n_broad):
    examples = []

    def add(msgs, tools):
        try:
            text = tokenizer.apply_chat_template(
                msgs, tools=tools, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            return
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=True, max_length=max_len)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        spans = unmask_spans(text)
        labels = [-100] * len(ids)
        si = 0
        for i, (a, b) in enumerate(offs):
            if a == b:
                continue
            while si < len(spans) and spans[si][1] <= a:
                si += 1
            if si < len(spans) and spans[si][0] <= a < spans[si][1]:
                labels[i] = ids[i]
        if any(x != -100 for x in labels):
            examples.append({"input_ids": ids, "labels": labels,
                             "attention_mask": [1] * len(ids)})

    swe = list(_iter(SWE))[:n_swe]
    for _ in range(swe_repeat):
        for row in swe:
            add(row["messages"], row.get("tools"))
    n = 0
    for row in _iter(BROAD):
        if n >= n_broad:
            break
        if row.get("half") not in (None, "mtp"):
            continue
        add(row["messages"], None)
        n += 1
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--n-swe", type=int, default=71)
    ap.add_argument("--swe-repeat", type=int, default=4)
    ap.add_argument("--n-broad", type=int, default=1200)
    args = ap.parse_args()

    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length,
        load_in_4bit=True, full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=args.lora_r, lora_alpha=args.lora_r,
        lora_dropout=0, bias="none", random_state=3407,
    )
    # gemma-4's FastModel tokenizer is a multimodal PROCESSOR — its __call__ can't
    # do return_offsets_mapping. Use a plain text AutoTokenizer (same vocab + the
    # base tool-aware chat template) for all data encoding.
    from transformers import AutoTokenizer
    text_tok = AutoTokenizer.from_pretrained(args.model)

    from datasets import Dataset
    exs = build_examples(text_tok, args.max_seq_length, args.n_swe, args.swe_repeat, args.n_broad)
    print(f"built {len(exs)} tokenized examples "
          f"(trained tokens: {sum(sum(1 for x in e['labels'] if x != -100) for e in exs)})",
          flush=True)
    ds = Dataset.from_list(exs)

    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=text_tok, padding=True),
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            max_steps=args.max_steps, learning_rate=args.lr, optim="adamw_8bit",
            lr_scheduler_type="linear", warmup_steps=5, logging_steps=10,
            output_dir=args.out, report_to="none",
            dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"tool-aware SFT adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
