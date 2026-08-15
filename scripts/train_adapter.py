#!/usr/bin/env python3
"""Train a PEFT LoRA or aLoRA adapter for a Granite causal LM.

Heavy dependencies are imported only after argument parsing so --help remains CPU-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="ibm-granite/granite-4.1-3b")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--technology", choices=("lora", "alora"), default="lora")
    parser.add_argument("--invocation", default="<pii_detection>")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated PEFT target module names",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("input"), str) or not isinstance(row.get("output"), str):
                raise ValueError(f"{path}:{line_number}: input and output must be strings")
            rows.append({"input": row["input"], "output": row["output"]})
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    invocation_tokens: list[int] | None = None
    if args.technology == "alora":
        invocation_tokens = tokenizer.encode(args.invocation, add_special_tokens=False)
        if not invocation_tokens:
            raise ValueError("The aLoRA invocation string tokenized to an empty sequence")
        first_character_tokens = tokenizer.encode(args.invocation[0], add_special_tokens=False)
        if (
            first_character_tokens != invocation_tokens[:1]
            or tokenizer.encode(args.invocation[1:], add_special_tokens=False)
            != invocation_tokens[1:]
        ):
            raise ValueError(
                "Granite Switch's current in-message aLoRA rewrite requires dropping the first "
                "character to preserve the remaining tokenization; choose a marker such as "
                "<pii_detection>."
            )

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    model.config.use_cache = False

    lora_kwargs = {
        "task_type": "CAUSAL_LM",
        "r": args.rank,
        "lora_alpha": args.alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": [item.strip() for item in args.target_modules.split(",") if item.strip()],
    }
    if invocation_tokens is not None:
        lora_kwargs["alora_invocation_tokens"] = invocation_tokens

    model = get_peft_model(model, LoraConfig(**lora_kwargs))
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    def encode(row: dict[str, str]) -> dict[str, list[int]]:
        user_content = row["input"]
        if args.technology == "alora":
            user_content = f"{user_content}\n{args.invocation}"
        user_message = {"role": "user", "content": user_content}
        prompt = tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        complete = tokenizer.apply_chat_template(
            [user_message, {"role": "assistant", "content": row["output"]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        complete_ids = tokenizer(
            complete,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )["input_ids"]
        if complete_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("The rendered assistant response does not preserve the prompt prefix")
        labels = [-100] * len(prompt_ids) + complete_ids[len(prompt_ids) :]
        if all(label == -100 for label in labels):
            raise ValueError("No assistant response tokens remain after truncation")
        attention_mask = [1] * len(complete_ids)
        padding = args.max_length - len(complete_ids)
        return {
            "input_ids": complete_ids + [tokenizer.pad_token_id] * padding,
            "attention_mask": attention_mask + [0] * padding,
            "labels": labels + [-100] * padding,
        }

    source_columns = ["input", "output"]
    train_dataset = Dataset.from_list(load_jsonl(args.train_file)).map(
        encode, remove_columns=source_columns
    )
    validation_dataset = Dataset.from_list(load_jsonl(args.validation_file)).map(
        encode, remove_columns=source_columns
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        load_best_model_at_end=True,
        report_to="none",
        seed=args.seed,
        bf16=bool(dtype == torch.bfloat16),
        fp16=bool(torch.cuda.is_available() and dtype is None),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved PEFT adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
