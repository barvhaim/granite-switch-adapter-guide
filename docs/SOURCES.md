# Sources

This documentation set is grounded in the following canonical and commit-pinned sources. The local Granite Switch repository was inspected at commit `91bf79900948adac0a62bd8b3514e36f766ed87b`. No benchmark results are asserted from these sources for the custom adapter described in this guide.

## Granite Switch

- Granite Switch repository at the inspected commit: https://github.com/generative-computing/granite-switch/tree/91bf79900948adac0a62bd8b3514e36f766ed87b
- Bring Your Own Adapter guide: https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/tutorials/guides/build_your_own_adapter.md
- Bring Your Own Adapter with Mellea guide: https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/tutorials/guides/mellea_build_your_own_adapter.md
- Composer reference: https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/src/granite_switch/composer/README.md
- Adapter discovery implementation: https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/src/granite_switch/composer/adapter_discovery.py
- Adapter loading and compatibility implementation: https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/src/granite_switch/composer/adapter_loader.py
- Chat-template and aLoRA invocation placement implementation: https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/src/granite_switch/composer/tokenizer_setup.py
- Hugging Face adapter invocation notebook: https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/tutorials/notebooks/hello_adapter.ipynb
- Direct Hugging Face adapter generation reference: https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/tutorials/scripts/reference/run_adapter_generation_direct.py
- Mellea plus vLLM adapter generation reference: https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/tutorials/scripts/reference/run_adapter_generation_mellea.py
- Upstream prerequisites: https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/tutorials/PREREQUISITES.md
- Granite Switch tests at the inspected commit: https://github.com/generative-computing/granite-switch/tree/91bf79900948adac0a62bd8b3514e36f766ed87b/tests

## Hugging Face PEFT, TRL, and Datasets

- PEFT LoRA and Activated LoRA reference: https://huggingface.co/docs/peft/main/en/package_reference/lora
- PEFT aLoRA fine-tuning example: https://github.com/huggingface/peft/tree/main/examples/alora_finetuning
- Activated LoRA paper: https://arxiv.org/abs/2504.12397
- TRL SFTTrainer: https://huggingface.co/docs/trl/main/en/sft_trainer
- TRL completion-only loss section: https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completion-only
- TRL assistant-only loss section: https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-assistant-messages-only
- TRL PEFT integration section: https://huggingface.co/docs/trl/main/en/sft_trainer#train-adapters-with-peft
- Hugging Face Datasets loading guide: https://huggingface.co/docs/datasets/loading
- PEFT checkpoint format guide: https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint

## Granite model and adapter libraries

- Granite 4.1 3B model: https://huggingface.co/ibm-granite/granite-4.1-3b
- Granite Libraries collection: https://huggingface.co/collections/ibm-granite/granite-libraries
- RAG adapter library: https://huggingface.co/ibm-granite/granitelib-rag-r1.0

## Inference and Mellea

- Transformers chat templates: https://huggingface.co/docs/transformers/main/en/chat_templating
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- Mellea LoRA and aLoRA adapters: https://docs.mellea.ai/advanced/lora-and-alora-adapters
- Mellea repository: https://github.com/generative-computing/mellea

## Interpretation note

The inspected Granite Switch commit contains a documentation inconsistency around custom Mellea adapters: the general BYOA guide includes a blanket unsupported note, while the dedicated Mellea BYOA guide demonstrates a lower-level path. Current Mellea documentation describes custom adapter training and loading. This guide therefore does not claim universal support or universal non-support. It recommends pinning versions and testing the exact Mellea backend, Granite Switch artifact, and vLLM combination.
