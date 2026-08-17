#!/usr/bin/env python3
"""Run a standalone PEFT adapter before Granite Switch composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="ibm-granite/granite-4.1-3b")
    parser.add_argument("--adapter", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt")
    inputs.add_argument(
        "--input-file", type=Path, help="JSONL rows with string id and input fields"
    )
    parser.add_argument("--output-file", type=Path, help="required with --input-file")
    parser.add_argument(
        "--invocation",
        help="Visible aLoRA marker, for example <pii_detection>",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.input_file is not None and args.output_file is None:
        parser.error("--output-file is required with --input-file")
    if args.prompt is not None and args.output_file is not None:
        parser.error("--output-file is only valid with --input-file")
    return args


def load_batch_inputs(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            row_id = value.get("id")
            prompt = value.get("input")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
            if not isinstance(prompt, str):
                raise ValueError(f"{path}:{line_number}: input must be a string")
            if row_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {row_id!r}")
            seen_ids.add(row_id)
            rows.append({"id": row_id, "input": prompt})
    if not rows:
        raise ValueError(f"{path}: no input rows")
    return rows


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def select_device_and_dtype(torch: Any) -> tuple[Any, Any]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda"), dtype
    if torch.backends.mps.is_available():
        # A 3B FP32 model is about 11.2 GiB before activations. FP32 is the
        # conservative compatibility choice for the documented 64 GB Mac path.
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = select_device_and_dtype(torch)
    print(f"Loading base model on {device} with dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    base.to(device)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    def generate(prompt: str) -> str:
        content = f"{prompt}\n{args.invocation}" if args.invocation else prompt
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    if args.prompt is not None:
        print(generate(args.prompt))
        return

    assert args.input_file is not None
    assert args.output_file is not None
    predictions = [
        {"id": row["id"], "output": generate(row["input"])}
        for row in load_batch_inputs(args.input_file)
    ]
    write_predictions(args.output_file, predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
