# 01 - Design the Adapter Contract

A useful Granite Switch adapter should implement one narrow function with a stable input and output contract. Do not begin with training. First decide what a caller can send, what the adapter must return, and how failures are represented.

Granite Switch is the composition and inference layer. It embeds already-trained LoRA or aLoRA weights into a Granite checkpoint and adds switching metadata, control tokens, and a chat template. Granite Switch composes adapters; it does not train adapters. Training belongs to PEFT, TRL, or another compatible training stack. See the [Granite Switch BYOA guide](https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/tutorials/guides/build_your_own_adapter.md) and the [composer reference](https://github.com/generative-computing/granite-switch/blob/91bf79900948adac0a62bd8b3514e36f766ed87b/src/granite_switch/composer/README.md).

## Start with one decision

For the included learning project, the function is:

- Input: one text string.
- Output: exactly one JSON object with `contains_pii` and `entities`.
- Meaning: identify whether the text contains an email, phone number, address, personal name, or another declared PII category.

This is intentionally narrower than "detect sensitive data." A narrow contract makes labels reviewable, evaluation deterministic, and regressions diagnosable. The sample data is synthetic and is only a teaching fixture, not a production PII corpus.

## Contract template

Record the following before creating data:

```text
Function name: pii_detection
Purpose: detect declared personally identifiable information categories
Inputs:
  text: non-empty UTF-8 string
Output JSON schema:
  type: object
  required: [contains_pii, entities]
  additionalProperties: false
  properties:
    contains_pii:
      type: boolean
    entities:
      type: array
      uniqueItems: true
      items:
        enum: [email, phone, address, name, other]
Error policy:
  malformed or missing input is rejected before model invocation
Decision boundary:
  contains_pii is true exactly when at least one declared entity appears
  entities contains each detected category once
Generation policy:
  temperature 0.0
  short token limit
  no prose outside JSON
```

## Define edge cases now

Write explicit policies for:

- Obfuscated email addresses and international phone formats.
- Names that are also ordinary words or organization names.
- Multiple PII categories in one text.
- Placeholder or obviously synthetic values.
- Indirect identifiers that do not fit the declared enum.
- Empty, truncated, non-English, or malformed inputs.

An annotator should be able to label each case without guessing your intent. If two reasonable reviewers often disagree, narrow or clarify the contract before adding examples.

## Separate the model contract from application policy

The adapter should emit the smallest useful semantic result. Application code can add retries, thresholds, logging, transformations, and user-facing messages. For example, keep a confidence threshold out of a boolean adapter unless the training labels actually express confidence.

The `io.yaml` file described later records the adapter name, output schema, optional instruction, generation parameters, and transformations. It does not replace a human-readable contract or a labeled evaluation set.

## Contract acceptance gate

Do not proceed until all of these are true:

- The function can be described in one sentence.
- Every input field has a type and validation rule.
- The output is machine-parseable and has no optional prose.
- Positive and negative examples exist for every important boundary.
- At least two people can label a small sample consistently, or one person can relabel it consistently after a delay.
- The same contract will be used for LoRA, aLoRA, Hugging Face, and vLLM evaluation.

Next: [02 - Build the JSONL Dataset](02-build-the-dataset.md).
