# 04 - Convert LoRA to aLoRA

After the LoRA baseline passes its quality gate, train an aLoRA version using the same task contract, data split, output parser, and evaluation harness. This isolates activation behavior from task behavior.

Activated LoRA applies adapter weights only from a chosen invocation sequence onward, leaving earlier tokens on the base model path so their KV cache can be reused. PEFT documents aLoRA and its constraints in the [LoRA reference](https://huggingface.co/docs/peft/main/en/package_reference/lora#activated-lora-alora), and provides an [aLoRA fine-tuning example](https://github.com/huggingface/peft/tree/main/examples/alora_finetuning). The design is described in the [Activated LoRA paper](https://arxiv.org/abs/2504.12397).

## The invocation sequence invariant

The exact same invocation sequence must be used in all four places:

1. The text rendered into every aLoRA training example at the activation boundary.
2. The integer token IDs stored as `alora_invocation_tokens` in `adapter_config.json`.
3. The prompt rendered for direct PEFT evaluation.
4. The composed Granite Switch chat template used by Hugging Face or vLLM inference.

The invariant is token-level, not merely visual. For tokenizer `T`, invocation text `S`, and configured IDs `I`:

```text
I == T.encode(S, add_special_tokens=False)
```

Also require:

- `S` appears after all reusable long context and immediately before adapter-specific prompting or the assistant boundary.
- The final intended occurrence of `S` is the activation point.
- `S` is never removed by truncation.
- Training loss includes response tokens after activation, not prompt tokens before activation.
- The same tokenizer revision and chat template are used in training, composition, and inference.

PEFT scans for the last occurrence of the configured invocation token sequence and adapts that sequence and later tokens. Granite Switch at the inspected commit reads `alora_invocation_tokens` from `adapter_config.json`, decodes them, and configures its chat template to place an adapter control token immediately before the invocation text. If the invocation corresponds to the assistant role boundary, the template uses the generation-prompt boundary fallback. See the pinned [tokenizer setup implementation](https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/src/granite_switch/composer/tokenizer_setup.py).

Do not hand-prepend Granite Switch's `<|adapter_name|>` control token to training data. That token is created during composition. Train against the PEFT invocation sequence; let the composed chat template perform runtime control-token placement.

## Choose a robust invocation string

A good invocation string:

- Uses tokens already in the base tokenizer when possible.
- Is unlikely to occur accidentally in user-controlled content.
- Has one stable tokenization with the pinned tokenizer.
- Is short but semantically clear.
- Sits after reusable context and before the response.

Example:

```python
invocation_string = "<pii_detection>"
invocation_ids = tokenizer.encode(invocation_string, add_special_tokens=False)
assert invocation_ids
assert tokenizer.encode(invocation_string[0], add_special_tokens=False) == invocation_ids[:1]
assert tokenizer.encode(invocation_string[1:], add_special_tokens=False) == invocation_ids[1:]
```

The second assertion is specific to the inspected Granite Switch in-message rewrite: its composed chat template replaces the first invocation token with the adapter control token and emits the remaining invocation text. A marker such as `<pii_detection>` should be verified against the exact tokenizer revision; do not assume every arbitrary string preserves its tail tokenization.

Add `alora_invocation_tokens=invocation_ids` to `LoraConfig`. PEFT notes that aLoRA is for causal language models, cannot be merged into the base model, and may need different capacity than LoRA. Treat rank as an experiment, not a guaranteed recommendation.

## Training delta from LoRA

Keep the LoRA recipe unchanged except for deliberate aLoRA changes:

```python
from peft import LoraConfig

invocation_string = "<pii_detection>"
invocation_ids = tokenizer.encode(invocation_string, add_special_tokens=False)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    alora_invocation_tokens=invocation_ids,
)
```

The displayed rank is illustrative. Preserve the frozen evaluation set and report the actual configuration used.

Stage the saved output here:

```text
outputs/pii-library/pii_detection/granite-4.1-3b/alora/
|-- adapter_config.json
|-- adapter_model.safetensors
`-- io.yaml
```

## Pre-compose invariant test

For every split, render and tokenize the final training prompt, then assert:

```python
positions = [
    i
    for i in range(len(input_ids) - len(invocation_ids) + 1)
    if input_ids[i : i + len(invocation_ids)] == invocation_ids
]
assert positions, "invocation sequence missing"
assert positions[-1] < first_response_token
assert all(label == -100 for label in labels[:first_response_token])
assert any(label != -100 for label in labels[first_response_token:])
```

After composition, render the same semantic request with `adapter_name="pii_detection"` and inspect token IDs around activation. Compare direct PEFT and composed Granite Switch outputs before performance benchmarking.

Next: [05 - Package the Adapter and io.yaml](05-package-the-adapter.md).
