#!/usr/bin/env python3
"""Run a standalone PEFT adapter before Granite Switch composition."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="ibm-granite/granite-4.1-3b")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--invocation",
        help="Prefix for an aLoRA adapter, for example <pii_detection>",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    content = f"{args.prompt}\n{args.invocation}" if args.invocation else args.prompt
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated = output[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
