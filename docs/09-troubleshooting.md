# 09 - Troubleshooting

Debug one boundary at a time: data, masking, PEFT adapter, package layout, composition, chat-template rendering, then backend generation. Do not start by changing rank or learning rate when the adapter may not be activating.

## Composer finds no adapters

Symptoms:

- `--list-adapters` returns nothing.
- Composition completes without the expected name.

Checks:

- Confirm the path is `pii_detection/granite-4.1-3b/lora/` or `.../alora/`.
- Confirm the final directory contains `io.yaml`, `adapter_config.json`, and `adapter_model.safetensors`.
- Confirm the target directory is named `granite-4.1-3b`, matching the target derived from the base model.
- If passing a single adapter directory, point directly at the directory containing `adapter_config.json`.
- Use `--technology-filter` only for a technology that actually exists.

The discovery rules are implemented in the pinned [adapter discovery module](https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/src/granite_switch/composer/adapter_discovery.py).

## Composer reports unknown or missing modules

Likely causes:

- Adapter trained on a different base model or revision.
- Wrong `target_modules` copied from another architecture.
- Incomplete or zero adapter weights.
- Config and safetensors came from different runs.

Actions:

- Compare `base_model_name_or_path`, architecture, and tokenizer identity.
- Inspect actual safetensors keys, not only `target_modules` in config.
- Read `compose_report.json` and do not waive unexpected or missing weights without a documented reason.
- Re-export from the selected best checkpoint into a clean final directory.

## `alora_invocation_tokens` is missing or empty

The composer requires non-empty `alora_invocation_tokens` for aLoRA and reads them from `adapter_config.json`. Recreate the config from the same `LoraConfig` used during training. Do not relabel an ordinary LoRA directory as `alora`.

## aLoRA quality collapses after composition

Check the invocation sequence invariant first:

```python
ids = tokenizer.encode(invocation_string, add_special_tokens=False)
assert ids == adapter_config["alora_invocation_tokens"]
```

Then inspect:

- Exact tokenizer revision.
- Rendered training, direct PEFT, composed HF, and vLLM token IDs.
- Whether truncation removed all or part of the invocation.
- Whether a later accidental occurrence became the last invocation.
- Whether the request bypassed `adapter_name` or `chat_template_kwargs`.
- Whether response-only labels begin after activation.

The pinned [tokenizer setup code](https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/src/granite_switch/composer/tokenizer_setup.py) documents Granite Switch's LoRA prefix and aLoRA invocation placement.

## Adapter output is identical to the base model

- Confirm the name is present in composed `config.json` and `adapter_index.json`.
- Pass `adapter_name` to Hugging Face `apply_chat_template`.
- Pass `extra_body={"chat_template_kwargs":{"adapter_name":"..."}}` through vLLM.
- Capture the rendered prompt and verify its control token.
- Compare logits or outputs on a known discriminating case.
- Verify the adapter weights contain non-zero LoRA B values after training.

## Output is JSON-like but invalid

- Confirm targets were canonical JSON without prose or Markdown fences.
- Use response-only loss and ensure all response tokens were labeled.
- Use deterministic decoding and a short completion limit.
- Validate every target against the same JSON Schema stored in `io.yaml`.
- Add malformed-output cases to evaluation; do not silently coerce arbitrary text.
- If using constrained decoding, test the exact backend configuration rather than assuming `io.yaml` enforces it automatically.

## Hugging Face and vLLM disagree

Control these variables before calling it a model bug:

- Same composed directory and file hashes.
- Same messages and adapter name.
- Same rendered chat-template semantics.
- Same max completion tokens, temperature, sampling, stop conditions, and tokenizer.
- Compatible dtypes and declared numeric tolerances.
- No server-side default template or generation config overriding the request.

Compare prompt tokens, then first-token logits on a small case, then parsed outputs across the frozen suite.

## CUDA out of memory or server startup failure

- Confirm GPU visibility and supported CUDA stack.
- Reduce concurrency, maximum model length, and GPU memory utilization carefully.
- Close stale model processes.
- Use the Hugging Face path for correctness debugging before adding server concurrency.
- Record every memory-related flag because it changes benchmark comparability.

The upstream [prerequisites](https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/tutorials/PREREQUISITES.md) describe hardware and installation expectations.

## Mellea does not find the custom adapter

Current Mellea documentation describes custom adapters, but backend paths differ. Check:

- Installed Mellea version matches the documentation version.
- The chosen backend supports the registration method you are using.
- The model ID or local path resolves from the Mellea process.
- Embedded Granite Switch adapters and direct PEFT adapters are not being confused.
- `io.yaml` is present and its name matches the registered intrinsic.

Use [Mellea's current adapter documentation](https://docs.mellea.ai/advanced/lora-and-alora-adapters), then validate the exact Mellea plus Granite Switch plus vLLM combination. Do not treat older blanket statements or newer direct-HF examples as proof for every deployment path.

Next: [10 - Production Checklist](10-production-checklist.md).
