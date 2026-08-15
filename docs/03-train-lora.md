# 03 - Train the LoRA Baseline

Train a normal LoRA first. It is the simplest way to validate the function, labels, prompt, target modules, and output parser before adding aLoRA's activation boundary.

Granite Switch composes adapters; it does not train them. Use PEFT with a trainer such as TRL. The [PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/package_reference/lora) explains `LoraConfig`, and the [TRL PEFT integration](https://huggingface.co/docs/trl/main/en/sft_trainer#train-adapters-with-peft) shows adapter training through `SFTTrainer`.

## Pin the training identity

Record at minimum:

- Base model: `ibm-granite/granite-4.1-3b`.
- Exact model revision.
- Tokenizer revision and chat template hash.
- PEFT, Transformers, TRL, Datasets, PyTorch, and CUDA versions.
- Dataset split hashes.
- Seed, precision, optimizer, scheduler, batch size, accumulation, epochs or steps, and sequence limit.
- LoRA rank, alpha, dropout, bias policy, and target modules.

All adapters composed together must be compatible with the same base model architecture. Do not select target modules by copying a configuration from another model family. Confirm module names on the loaded Granite model and inspect the saved adapter's weight keys.

## Minimal training skeleton

This is a reference skeleton, not a claim that it was executed in this repository:

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

base_model = "ibm-granite/granite-4.1-3b"
dataset = load_dataset(
    "json",
    data_files={
        "train": "examples/pii_detection/train.jsonl",
        "validation": "examples/pii_detection/validation.jsonl",
    },
)


def to_prompt_completion(row):
    return {
        "prompt": [{"role": "user", "content": row["input"]}],
        "completion": [{"role": "assistant", "content": row["output"]}],
    }


dataset = dataset.map(
    to_prompt_completion,
    remove_columns=dataset["train"].column_names,
)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

args = SFTConfig(
    output_dir="artifacts/checkpoints/pii_detection-lora",
    completion_only_loss=True,
    max_length=2048,
    eval_strategy="steps",
    save_strategy="steps",
    seed=17,
)

trainer = SFTTrainer(
    model=base_model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
)
trainer.train()
trainer.save_model("outputs/pii-library/pii_detection/granite-4.1-3b/lora")
```

Treat all shown hyperparameters, including rank and target modules, as starting hypotheses. Confirm them against the exact model revision and tune them using validation data. Do not treat training loss alone as task quality.

The repository also includes a `transformers.Trainer` implementation with explicit response-token masking. After `uv sync --extra train`, run:

```bash
uv run python scripts/train_adapter.py \
  --technology lora \
  --base-model ibm-granite/granite-4.1-3b \
  --train-file examples/pii_detection/train.jsonl \
  --validation-file examples/pii_detection/validation.jsonl \
  --output-dir outputs/pii-library/pii_detection/granite-4.1-3b/lora \
  --gradient-checkpointing
```

This is a GPU training path. CI verifies the script interface and helper logic, but does not download or train the model.

## Required PEFT output

After `save_pretrained` or `save_model`, the final composer input must be staged as:

```text
outputs/pii-library/
`-- pii_detection/
    `-- granite-4.1-3b/
        `-- lora/
            |-- adapter_config.json
            |-- adapter_model.safetensors
            `-- io.yaml
```

PEFT creates `adapter_config.json` and adapter weights. You author `io.yaml` separately. Prefer `adapter_model.safetensors`; the upstream composer can also load a PyTorch `.bin`, but its library discovery path checks for the safetensors filename.

## Baseline acceptance gate

Before moving to aLoRA:

- Load the base model plus the saved PEFT adapter outside Granite Switch.
- Run the frozen test set with deterministic decoding.
- Measure schema validity, exact match or task metric, class-specific errors, and latency separately.
- Inspect failures and update data or contract only through a documented iteration.
- Save representative prompts, raw completions, parsed outputs, and environment metadata.
- Verify the adapter does not silently depend on information omitted from the contract.

Only proceed when the LoRA baseline is good enough to serve as a reference. aLoRA changes activation and cache reuse, not the task definition.

Next: [04 - Convert LoRA to aLoRA](04-train-alora.md).
