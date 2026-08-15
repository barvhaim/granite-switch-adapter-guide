#!/usr/bin/env python3
"""Invoke an embedded adapter in a composed Granite Switch model."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-name", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--invocation",
        help="Append an aLoRA invocation marker, for example <pii_detection>",
    )
    parser.add_argument("--backend", choices=("hf", "openai"), default="hf")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def user_content(args: argparse.Namespace) -> str:
    if args.invocation:
        return f"{args.prompt}\n{args.invocation}"
    return args.prompt


def run_hf(args: argparse.Namespace) -> None:
    import granite_switch.hf  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content(args)}],
        tokenize=False,
        add_generation_prompt=True,
        adapter_name=args.adapter_name,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated = output[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True))


def run_openai(args: argparse.Namespace) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="unused")
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": user_content(args)}],
        extra_body={"chat_template_kwargs": {"adapter_name": args.adapter_name}},
        max_completion_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    print(response.choices[0].message.content)


def main() -> None:
    args = parse_args()
    if args.backend == "hf":
        run_hf(args)
    else:
        run_openai(args)


if __name__ == "__main__":
    main()
