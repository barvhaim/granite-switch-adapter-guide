# 02 - Build the JSONL Dataset

Use JSON Lines so each line is one independent JSON value. Keep train, validation, and test splits separate. The [Hugging Face Datasets loading guide](https://huggingface.co/docs/datasets/loading) documents loading local JSON and JSONL data.

## Repository record shape

The executable scripts and validators in this repository use a deliberately small source format:

```json
{"id":"train-001","input":"Email the receipt to alex@example.com.","output":"{\"contains_pii\":true,\"entities\":[\"email\"]}"}
{"id":"train-002","input":"The deployment finished successfully.","output":"{\"contains_pii\":false,\"entities\":[]}"}
```

The training path transforms each record into a conversational prompt-completion example before tokenization. If you use TRL directly, convert `input` to a user `prompt` and `output` to an assistant `completion`. TRL computes completion-only loss by default for prompt-completion datasets; this can be made explicit with `completion_only_loss=True`. See the [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completion-only).

## Response-only loss is a contract requirement

The prompt is context, not a target to imitate. Mask prompt tokens and compute loss only on the assistant completion. Otherwise the adapter spends capacity learning to reproduce system and user text, and prompt wording can dominate the short structured answer.

Use one of these supported approaches:

- Conversational prompt-completion records with `completion_only_loss=True`.
- Conversational records with `assistant_only_loss=True` when the selected chat template exposes the generation mask required by TRL.
- A custom data collator that sets all non-response labels to `-100`, with an automated masking test.

Do not assume a flag worked. Tokenize one positive and one negative record and verify:

- Prompt token labels are all `-100`.
- Every intended assistant output token contributes to loss.
- EOS contributes if termination is part of the target.
- Padding and tokens after padding do not contribute.

## Data quality rules

- Emit canonical JSON with the same key names and value types in every target.
- Include both classes and inspect class balance.
- Deduplicate before splitting, including near-duplicates and templated variants.
- Split by source entity or scenario when related examples could leak across splits.
- Keep the test set frozen and out of training-time prompt iteration.
- Record provenance, license, annotator policy, and redaction decisions outside the model input.
- Remove secrets and personal, confidential, or proprietary content.
- Preserve hard negatives, not only obvious negatives.
- Keep inputs within the eventual serving context limit.

## aLoRA-ready data design

Even though the recommended sequence is LoRA first and aLoRA second, design prompts so an invocation boundary can be added later without changing the task. For example:

```text
Email the receipt to alex@example.com.
<pii_detection>
```

`<pii_detection>` is an illustrative invocation string, not a new tokenizer token. For aLoRA, the exact tokenizer IDs for the chosen string become `alora_invocation_tokens`, and the string must occur at the activation boundary in every training and inference prompt. The invariant is covered in [04 - Convert LoRA to aLoRA](04-train-alora.md).

Do not include the aLoRA invocation string in the completion. It belongs at the end of adapter-specific prompting and before the answer tokens.

## Split and validation checklist

For each split, validate mechanically:

- Every line parses as one JSON object.
- `id` values are unique across train, validation, and test.
- `input` and `output` are non-empty strings.
- Each `output` string parses against the output JSON Schema.
- No unknown output keys are present.
- The chosen invocation string is absent for the initial LoRA baseline, or present exactly where expected for aLoRA.
- No normalized input or output duplicates cross split boundaries.
- Per-class counts and input token-length percentiles are reported without inventing quality claims.

The repository does not include benchmark results for your custom data. Establish your own frozen baseline rather than copying numbers from unrelated adapters.

Next: [03 - Train the LoRA Baseline](03-train-lora.md).
