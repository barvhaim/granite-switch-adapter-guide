# 06 - Compose the Granite Switch Checkpoint

Composition embeds trained adapter weights into a Granite Switch checkpoint and generates the switching assets. It is a build step, not a training step. Granite Switch composes adapters; it does not train adapters.

Install and command details are documented in the pinned [Granite Switch composer reference](https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/src/granite_switch/composer/README.md).

## Install the composer

```bash
python -m pip install "granite-switch[compose]"
```

For a source checkout, follow the upstream prerequisites and pin the checkout or package version used for the build.

## Inspect before building

List adapters found in the local library:

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ./outputs/pii-library \
  --list-adapters
```

A single technology directory can also be passed directly:

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ./outputs/pii-library/pii_detection/granite-4.1-3b/alora \
  --list-adapters
```

Review the discovered adapter name and technology. An empty list usually means the tree, target model directory, `io.yaml`, config, or safetensors filename is wrong.

## Compose the LoRA reference

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ./outputs/pii-library \
  --include-adapters pii_detection \
  --technology-filter lora \
  --output ./outputs/composed/pii-detection-lora
```

## Compose the aLoRA candidate

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ./outputs/pii-library \
  --include-adapters pii_detection \
  --technology-filter alora \
  --output ./outputs/composed/pii-detection-alora
```

Build the two technologies into separate checkpoints for parity testing because duplicate adapter names are deduplicated and aLoRA is preferred when both are discovered without a filter.

## Compose with other adapters

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ./outputs/pii-library \
             ibm-granite/granitelib-rag-r1.0 \
  --output ./outputs/composed/custom-plus-rag
```

All included adapters must target the compatible base architecture. Ranks and alpha values may differ: the inspected composer allocates to the maximum rank, zero-pads smaller adapters, and stores per-adapter scaling. Compatibility still must be validated from the actual saved modules and weights.

## Expected output

A composed directory should include model shards plus assets such as:

```text
config.json
model-*.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.jinja
adapter_index.json
compose_report.json
BUILD.md
io_configs/pii_detection/io.yaml
```

Exact shard counts depend on the build. Do not encode a fixed count in deployment checks.

## Inspect the build artifacts

Before inference:

- Confirm `config.json` lists `pii_detection` in `adapter_names`.
- Confirm `adapter_index.json` maps the adapter to one control token and copied `io.yaml`.
- Confirm `io_configs/pii_detection/io.yaml` matches the approved source.
- Read `compose_report.json` for missing, zero, unexpected, or not-targeted module findings.
- Read `BUILD.md` and record base source, adapter sources, and build parameters.
- Load tokenizer assets and render one request with `adapter_name`.
- For aLoRA, inspect the activation location against the invocation invariant.
- Hash and archive the complete composed directory after verification.

The command examples are documented procedures. This guide does not claim they were executed on your hardware or against your private adapter artifacts.

Next: [07 - Invoke with Hugging Face and vLLM](07-inference.md).
