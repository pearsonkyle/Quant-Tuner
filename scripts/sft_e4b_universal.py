#!/usr/bin/env python3
"""Universal-corpus QLoRA SFT of gemma-4-E4B — tool calls + reasoning + breadth.

Trains on data/sft.jsonl.gz (curated: logs, logs-agents, swe-trajectories,
redteam-refusals, broad-instruct; with tool_calls + reasoning_content and
pre-scrubbed system prompts). Reuses the tool-call-preserving recipe from
sft_e4b_tools.py (base gemma-4 template + custom masking) and adds:

  - reasoning merge: this data stores reasoning as a SEPARATE assistant message
    ({role:assistant, reasoning_content:...}) before the action. The gemma-4
    template only renders reasoning_content when it's on the SAME message as the
    content/tool_calls, so merge_reasoning() attaches a reasoning-only message to
    the following assistant action. Then it renders as
    <|channel>thought ... <channel| > <|tool_call>call:... — and the custom mask
    trains the thought channel + tool_calls, masks <|tool_response> + prompts.
  - 8k context (--max-seq-length 8192) so long agentic trajectories fit.
"""
from __future__ import annotations

import argparse
import gzip
import json

from sft_e4b_tools import unmask_spans  # same masking (thought channel is inside model turn)


def merge_reasoning(messages: list[dict]) -> list[dict]:
    """Attach a reasoning-only assistant message to the following assistant action
    so the template renders its <|channel>thought block."""
    out: list[dict] = []
    pending_reasoning = None
    for m in messages:
        is_reason_only = (
            m.get("role") == "assistant"
            and m.get("reasoning_content")
            and not m.get("content")
            and not m.get("tool_calls")
        )
        if is_reason_only:
            pending_reasoning = (pending_reasoning or "") + m["reasoning_content"]
            continue
        if pending_reasoning and m.get("role") == "assistant":
            m = {**m, "reasoning_content": pending_reasoning + (m.get("reasoning_content") or "")}
            pending_reasoning = None
        elif pending_reasoning:
            # reasoning not followed by an assistant action — emit it as its own turn
            out.append({"role": "assistant", "reasoning_content": pending_reasoning, "content": ""})
            pending_reasoning = None
        out.append(m)
    if pending_reasoning:
        out.append({"role": "assistant", "reasoning_content": pending_reasoning, "content": ""})
    return out


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _tokenize_example(tokenizer, msgs, tools, max_len):
    """Render msgs (base template + tools) and build masked labels via unmask_spans.
    The gemma-4 template renders reasoning only for the FINAL assistant turn, so
    callers pass truncated msg lists ending at a reasoning turn to train it."""
    try:
        text = tokenizer.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return None
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
    if not any(x != -100 for x in labels):
        return None
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def build_examples(tokenizer, data_path, max_len, split, max_rows):
    examples = []
    n = 0
    with _open(data_path) as f:
        for line in f:
            if n >= max_rows:
                break
            d = json.loads(line)
            if d.get("split") != split:
                continue
            merged = merge_reasoning(d["messages"])
            tools = d.get("tools")
            # whole conversation: trains all tool_calls + the final-turn reasoning
            ex = _tokenize_example(tokenizer, merged, tools, max_len)
            if ex:
                examples.append(ex)
                n += 1
            # reasoning turns are stripped from history by the template, so add a
            # truncated example ending at EACH reasoning-bearing assistant turn so
            # its <|channel>thought renders and trains.
            if d.get("n_reasoning", 0) > 0:
                for k, m in enumerate(merged):
                    if k == len(merged) - 1:
                        continue  # already covered by the whole conversation
                    if m.get("role") == "assistant" and m.get("reasoning_content"):
                        ex = _tokenize_example(tokenizer, merged[: k + 1], tools, max_len)
                        if ex:
                            examples.append(ex)
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/sft.jsonl.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-rows", type=int, default=100000)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    args = ap.parse_args()

    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length,
        load_in_4bit=True, full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=args.lora_r, lora_alpha=args.lora_r,
        lora_dropout=0, bias="none", random_state=3407, use_gradient_checkpointing="unsloth",
    )
    from transformers import AutoTokenizer
    text_tok = AutoTokenizer.from_pretrained(args.model)

    from datasets import Dataset
    exs = build_examples(text_tok, args.data, args.max_seq_length, args.split, args.max_rows)
    trained = sum(sum(1 for x in e["labels"] if x != -100) for e in exs)
    print(f"built {len(exs)} tokenized examples (trained tokens: {trained})", flush=True)
    ds = Dataset.from_list(exs)

    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=text_tok, padding=True),
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            max_steps=args.max_steps, learning_rate=args.lr, optim="adamw_8bit",
            lr_scheduler_type="linear", warmup_steps=10, logging_steps=10,
            save_steps=100, save_total_limit=2,
            output_dir=args.out, report_to="none",
            dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"universal SFT adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
