# Granite Switch Custom Adapter Guide

An executable, source-grounded guide to designing, training, packaging, composing, serving, and evaluating a custom LoRA or activated LoRA (aLoRA) adapter for [Granite Switch](https://github.com/generative-computing/granite-switch).

> Granite Switch composes already-trained PEFT adapters into one deployable checkpoint. It does not train the adapters for you.

## What is verified here

The CPU-only helper package, validators, examples, tests, and command generation run in CI without downloading model weights. The training and inference scripts are complete reference paths, but require a suitable GPU and are not presented as having been trained by this repository's CI.

The upstream behavior was inspected at Granite Switch commit [`91bf79900948adac0a62bd8b3514e36f766ed87b`](https://github.com/generative-computing/granite-switch/commit/91bf79900948adac0a62bd8b3514e36f766ed87b). See [sources and provenance](docs/SOURCES.md).

## The complete path

```text
function contract
    -> JSONL train/validation data
    -> PEFT LoRA or aLoRA training
    -> standalone adapter evaluation
    -> library-style adapter directory + io.yaml
    -> Granite Switch Composer
    -> Hugging Face or vLLM inference
    -> quality, schema, and backend-parity gates
```

## 10-minute CPU-only first lab

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), no GPU, no credentials, no model download.

```bash
git clone https://github.com/njs2017/granite-switch-adapter-guide.git
cd granite-switch-adapter-guide
uv sync --group dev

uv run granite-adapter-guide validate-dataset \
  examples/pii_detection/train.jsonl

uv run granite-adapter-guide evaluate \
  examples/pii_detection/validation.jsonl \
  examples/pii_detection/predictions.example.jsonl \
  --schema examples/pii_detection/schema.json

uv run python scripts/compose_model.py \
  --adapter ./outputs/pii-library/pii_detection/granite-4.1-3b/alora \
  --output ./composed-model
```

The final command prints the exact Composer invocation without running a model build. Add `--execute` only after installing the compose dependencies and producing real adapter weights.

## Guided learning path

1. [Design a narrow function contract](docs/01-design-the-contract.md)
2. [Build and split the dataset](docs/02-build-the-dataset.md)
3. [Train a LoRA baseline](docs/03-train-lora.md)
4. [Train aLoRA with a stable invocation sequence](docs/04-train-alora.md)
5. [Package the PEFT output and io.yaml](docs/05-package-the-adapter.md)
6. [Compose a Granite Switch checkpoint](docs/06-compose-granite-switch.md)
7. [Run Hugging Face and vLLM inference](docs/07-inference.md)
8. [Evaluate quality, schema, and parity](docs/08-evaluation.md)
9. [Troubleshoot common failures](docs/09-troubleshooting.md)
10. [Apply production acceptance gates](docs/10-production-checklist.md)

## Example project: `pii_detection`

The included synthetic example defines a narrow JSON-producing function:

```json
{
  "contains_pii": true,
  "entities": ["email"]
}
```

Validate its data:

```bash
uv run granite-adapter-guide validate-dataset \
  examples/pii_detection/train.jsonl \
  examples/pii_detection/validation.jsonl
```

### Train LoRA first

Install the optional training environment:

```bash
uv sync --extra train
```

Run on a suitable CUDA system:

```bash
uv run python scripts/train_adapter.py \
  --technology lora \
  --base-model ibm-granite/granite-4.1-3b \
  --train-file examples/pii_detection/train.jsonl \
  --validation-file examples/pii_detection/validation.jsonl \
  --output-dir outputs/pii-library/pii_detection/granite-4.1-3b/lora \
  --gradient-checkpointing
```

The script masks prompt tokens with `-100`, so training loss is computed only on the assistant response.

### Move to aLoRA

Use a stable invocation sequence included in every training prompt:

```bash
uv run python scripts/train_adapter.py \
  --technology alora \
  --invocation '<pii_detection>' \
  --base-model ibm-granite/granite-4.1-3b \
  --train-file examples/pii_detection/train.jsonl \
  --validation-file examples/pii_detection/validation.jsonl \
  --output-dir outputs/pii-library/pii_detection/granite-4.1-3b/alora \
  --gradient-checkpointing
```

For aLoRA, PEFT writes `alora_invocation_tokens` into `adapter_config.json`. Granite Switch reads this field while building the embedded control-token path.

Copy the function contract next to the weights:

```bash
cp examples/pii_detection/io.yaml \
  outputs/pii-library/pii_detection/granite-4.1-3b/alora/io.yaml
```

Validate the resulting adapter directory:

```bash
uv run granite-adapter-guide validate-adapter \
  outputs/pii-library/pii_detection/granite-4.1-3b/alora
```

## Compose the checkpoint

```bash
uv sync --extra compose

uv run python scripts/compose_model.py \
  --base-model ibm-granite/granite-4.1-3b \
  --adapter outputs/pii-library/pii_detection/granite-4.1-3b/alora \
  --output composed-model \
  --execute
```

Mix the custom adapter with an official library by repeating `--with-library`:

```bash
uv run python scripts/compose_model.py \
  --adapter outputs/pii-library/pii_detection/granite-4.1-3b/alora \
  --with-library ibm-granite/granitelib-rag-r1.0 \
  --output composed-model \
  --execute
```

## Invoke the embedded adapter

Hugging Face:

```bash
uv run python scripts/run_composed_model.py \
  --backend hf \
  --model ./composed-model \
  --adapter-name pii_detection \
  --invocation '<pii_detection>' \
  --prompt 'Contact me at alex@example.com'
```

vLLM server:

```bash
uv sync --extra serve
uv run vllm serve ./composed-model --port 8000
```

OpenAI-compatible client:

```bash
uv run python scripts/run_composed_model.py \
  --backend openai \
  --model ./composed-model \
  --adapter-name pii_detection \
  --invocation '<pii_detection>' \
  --prompt 'Contact me at alex@example.com'
```

Both paths pass `adapter_name` through the composed chat template rather than manually prepending the internal `<|pii_detection|>` control token. The example's aLoRA was trained with the visible `<pii_detection>` invocation sequence, so `--invocation` appends that same marker to the user message. Omit `--invocation` for ordinary LoRA or for an aLoRA trained on the model's assistant-role boundary.

## Repository layout

```text
.
├── docs/                       detailed, sequential guide
├── examples/pii_detection/     synthetic data and io.yaml contract
├── scripts/                    training, composition, and inference paths
├── src/granite_adapter_guide/  CPU-only validation and evaluation CLI
├── tests/                      deterministic regression tests
└── .github/workflows/ci.yml    lint, tests, CLI smoke test, compile check
```

## Key invariants

- Every adapter in one composed checkpoint must target the same base model.
- The directory name must identify the target model and technology: `<name>/granite-4.1-3b/{lora|alora}/`.
- aLoRA requires nonempty `alora_invocation_tokens` in `adapter_config.json`.
- The invocation sequence used for aLoRA training must appear at the same semantic boundary during inference.
- Adapter quality must be measured before and after composition.
- A valid JSON schema is not evidence that the predicted values are correct.

## Development

```bash
uv sync --group dev
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
uv run python -m compileall -q src scripts
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change.

## License

Apache-2.0. See [LICENSE](LICENSE).
