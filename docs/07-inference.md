# 07 - Invoke with Hugging Face and vLLM

Always select an embedded adapter through the composed chat template. Do not manually guess control-token placement. The template distinguishes LoRA prefix activation from aLoRA activation at its invocation sequence.

The examples below target the included marker-based aLoRA recipe, so the user message ends with `<pii_detection>`, exactly as it did during training. For ordinary LoRA, omit that visible invocation marker. For an aLoRA trained on the model's assistant-role boundary instead of a visible marker, preserve that exact boundary rather than adding `<pii_detection>`. The examples intentionally do not add a new system message: prompt parity requires the same message shape used during training unless a changed prompt has been evaluated separately.

## Hugging Face

Import the Granite Switch Hugging Face backend before using `AutoModelForCausalLM` so the custom architecture is registered.

```python
import json
import torch
import granite_switch.hf
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./outputs/composed/pii-detection-alora"
adapter_name = "pii_detection"

model = AutoModelForCausalLM.from_pretrained(model_path).eval().to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": "Email the receipt to alex@example.com.\n<pii_detection>",
    },
]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
    adapter_name=adapter_name,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
result = json.loads(raw)
assert set(result) == {"contains_pii", "entities"}
assert isinstance(result["contains_pii"], bool)
assert isinstance(result["entities"], list)
print(result)
```

The activation API is `adapter_name=...` on `tokenizer.apply_chat_template`, as shown by the upstream [Hugging Face adapter notebook](https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/tutorials/notebooks/hello_adapter.ipynb).

## vLLM server

Install the compatible Granite Switch vLLM extra and start the OpenAI-compatible server. In this repository's environment:

```bash
uv sync --extra serve
uv run vllm serve ./outputs/composed/pii-detection-alora --port 8000
```

The `serve` extra follows Granite Switch's default vLLM line. If your CUDA stack requires the upstream `vllm20` extra, follow the version note in Granite Switch's README instead of forcing an incompatible runtime.

Send the adapter selection in `chat_template_kwargs`:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="./outputs/composed/pii-detection-alora",
    messages=[
        {
            "role": "user",
            "content": "Email the receipt to alex@example.com.\n<pii_detection>",
        },
    ],
    extra_body={"chat_template_kwargs": {"adapter_name": "pii_detection"}},
    temperature=0.0,
    max_completion_tokens=100,
)

raw = response.choices[0].message.content
result = json.loads(raw)
print(result)
```

This `chat_template_kwargs` pattern is shown in the pinned [Granite Switch BYOA guide](https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/tutorials/guides/build_your_own_adapter.md).

## Do not bypass the template

Manual control-token concatenation is fragile because:

- LoRA and aLoRA use different placement rules.
- aLoRA activation depends on its stored invocation sequence.
- The inspected composer performs token-exchange handling to avoid duplicating the first substituted embedding at activation.
- A chat-template change can alter role-boundary tokens.

For tests, capture the rendered prompt and token IDs. In production, keep the public request at the message level and pass only the approved adapter name.

## Current Mellea nuance

The Granite Switch checkout at the inspected commit contains mixed guidance: one BYOA page says custom adapters are unsupported by Mellea, while a separate guide demonstrates a lower-level custom intrinsic path. Current Mellea documentation now describes training, uploading, and loading custom LoRA or aLoRA adapters with `CustomIntrinsicAdapter`; see [Mellea LoRA and aLoRA adapters](https://docs.mellea.ai/advanced/lora-and-alora-adapters).

Do not infer from that page that every custom adapter, Mellea backend, and composed Granite Switch deployment is interchangeable. Mellea's documented example may use a direct local Hugging Face backend, while Granite Switch composition is commonly served through vLLM. Pin Mellea and Granite Switch versions, confirm that the selected backend loads embedded adapter metadata and `io.yaml`, and run the same parity suite. For the narrowest and most transparent learning path, use direct Hugging Face and vLLM invocation first.

Next: [08 - Evaluate Quality and Backend Parity](08-evaluation.md).
