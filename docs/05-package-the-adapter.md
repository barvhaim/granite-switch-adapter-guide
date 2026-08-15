# 05 - Package the Adapter and io.yaml

The Granite Switch composer accepts a single adapter directory or discovers adapters in a library-style tree. Use the standard tree even for one adapter because it makes the name, target model, and technology explicit.

## Required layout

Use exactly this shape:

```text
adapter-library/
`-- pii_detection/
    `-- granite-4.1-3b/
        |-- lora/
        |   |-- adapter_config.json
        |   |-- adapter_model.safetensors
        |   `-- io.yaml
        `-- alora/
            |-- adapter_config.json
            |-- adapter_model.safetensors
            `-- io.yaml
```

In symbolic form:

```text
<adapter_name>/granite-4.1-3b/{lora|alora}/
```

The upstream discovery code searches for `adapter_name/model/technology/io.yaml`, accepts `lora` and `alora`, and requires both `adapter_config.json` and `adapter_model.safetensors`. If both technologies exist under one discovered adapter name, aLoRA is preferred unless a technology filter is supplied. See the pinned [adapter discovery implementation](https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/src/granite_switch/composer/adapter_discovery.py).

The adapter name used by discovery comes from the directory tree. Keep it identical to the `name` in `io.yaml` to avoid operational confusion.

## Minimal io.yaml

```yaml
name: pii_detection
```

Technology is not an `io.yaml` field. It is inferred from the final `lora/` or `alora/` directory.

## Recommended io.yaml

```yaml
name: pii_detection
model: ~
response_format: |
  {
    "type": "object",
    "properties": {
      "contains_pii": {
        "type": "boolean"
      },
      "entities": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["email", "phone", "address", "name", "other"]
        },
        "uniqueItems": true
      }
    },
    "required": ["contains_pii", "entities"],
    "additionalProperties": false
  }
transformations: []
instruction: |
  Detect declared PII categories in the input. Return only the JSON object
  required by response_format.
parameters:
  max_completion_tokens: 100
  temperature: 0.0
sentence_boundaries: ~
```

Granite Switch uses `io.yaml` for discovery and copies it into the composed checkpoint. Fields beyond `name` are useful to higher-level runtimes for structured output, preprocessing, post-processing, or generation settings. Confirm the exact consumer semantics in the runtime version you deploy; a copied schema does not automatically enforce constrained decoding in every direct Hugging Face call.

The upstream [BYOA guide](https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/tutorials/guides/build_your_own_adapter.md) shows `response_format`, `transformations`, `instruction`, `parameters`, and `sentence_boundaries` in a production-style example.

## Validate adapter_config.json

For both technologies, verify:

- `base_model_name_or_path` identifies the intended Granite base model or is documented if rewritten by tooling.
- `task_type` is `CAUSAL_LM`.
- `r`, `lora_alpha`, `target_modules`, and PEFT type match the run record.
- Saved weight keys correspond to declared target modules.

For aLoRA also verify:

- `alora_invocation_tokens` exists and is a non-empty list of integers.
- Decoding and re-encoding with the pinned tokenizer gives the intended sequence.
- The sequence occurs at the tested runtime activation boundary.

## Package manifest

Alongside, but not inside the three composer-required files, maintain a release manifest with:

- Base model ID and revision.
- Tokenizer revision and chat template hash.
- Training code revision and dependency lock hash.
- Train, validation, and test dataset hashes.
- Adapter file hashes.
- Contract and `io.yaml` schema version.
- Evaluation report path and approval status.

Do not place optimizer states, full base-model weights, arbitrary checkpoints, or secrets in the final adapter technology directory.

Next: [06 - Compose the Granite Switch Checkpoint](06-compose-granite-switch.md).
