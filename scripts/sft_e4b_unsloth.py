#!/usr/bin/env python3
"""QLoRA SFT of gemma-4-E4B (the target) with Unsloth, on our messages JSONL.

Purpose (target-side lever): teach the target the off-policy data it CAN learn
(agentic logs + FinePhrase) so a later on-policy distillation passes it to the
drafter. QLoRA 4-bit + seq<=4096 to fit a 16 GB card.

    python scripts/sft_e4b_unsloth.py --model ~/Programs/llm/hf/gemma-4-E4B-it \
        --data out/sft/agentic-logs.jsonl --out out/sft/e4b-agentic \
        --max-seq-length 2048 --max-steps 60
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()

    from unsloth import FastModel  # noqa: E402  (must import before transformers)
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    from datasets import Dataset

    with open(args.data) as _f:
        rows = [json.loads(line) for line in _f]
    ds = Dataset.from_list(rows)

    def fmt(examples):
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
            for c in examples["messages"]
        ]
        return {"text": texts}

    ds = ds.map(fmt, batched=True)

    from trl import SFTConfig, SFTTrainer

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            warmup_steps=5,
            logging_steps=5,
            output_dir=args.out,
            report_to="none",
        ),
    )
    # mask loss to assistant responses only (gemma-4 turn markers)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"SFT LoRA adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
