# 08 - Evaluate Quality and Backend Parity

Evaluation has three separate questions:

1. Does the adapter implement the contract?
2. Does aLoRA preserve acceptable task behavior relative to LoRA?
3. Do direct PEFT, composed Hugging Face, and composed vLLM agree closely enough for deployment?

Do not replace these questions with training loss or an unrelated throughput benchmark. Do not fabricate benchmark numbers. This repository intentionally provides a method, not results for an adapter that has not been trained here.

## Freeze an evaluation matrix

Evaluate the same test case IDs across:

- Base model with no adapter, as a sanity control.
- Direct PEFT LoRA.
- Composed Granite Switch LoRA through Hugging Face.
- Direct PEFT aLoRA.
- Composed Granite Switch aLoRA through Hugging Face.
- Composed Granite Switch aLoRA through vLLM.

Use identical semantic messages, tokenizer revision, generation limits, deterministic decoding, and output parser. Capture the rendered prompts and token IDs for each path because template differences are often the real source of output differences.

## Contract metrics

For a boolean JSON function, report at least:

- JSON parse rate.
- JSON Schema validity rate.
- Exact output match.
- Accuracy and a confusion matrix.
- Precision, recall, and F1 for the operationally important class.
- Error counts by documented edge-case category.
- Empty, truncated, or extra-prose output rate.

For other contracts, choose task-appropriate metrics and always retain schema validity and failure rates. Report numerator, denominator, split identity, and confidence intervals when useful. Never publish only an aggregate score if one class can hide another.

## Parity levels

Use increasingly tolerant parity checks:

### 1. Prompt parity

- Same user-visible input fields and values.
- Same chat-template revision.
- Adapter control token at the expected LoRA or aLoRA location.
- For aLoRA, same configured invocation token IDs and activation boundary.

### 2. Logit parity

When practical, compare next-token logits for a small deterministic sample using the same dtype and device policy. Use explicit numeric tolerances; bitwise identity is not generally a portable requirement across kernels and hardware.

### 3. Output parity

Compare parsed structured outputs, not only raw strings. Require exact semantic agreement for deterministic classifiers. Keep raw differences for diagnosis.

### 4. Metric parity

Run the complete frozen test set and compare all contract metrics against predeclared release tolerances. A matching aggregate with different high-severity failures is not parity.

The upstream Granite Switch repository includes its own HF, vLLM, generation-equivalence, and integration test areas, but those validate the project implementation rather than your custom adapter's task quality. See the pinned [test tree](https://github.com/generative-computing/granite-switch/tree/91bf79900948adac0a62bd8b3514e36f766ed87b/tests).

## aLoRA-specific tests

Include cases where:

- The invocation string occurs exactly once at the intended boundary.
- Similar text appears in user content but must not become the final activation point.
- Long context approaches the truncation limit.
- The invocation is missing, altered, or split, and the harness rejects the request before measuring task quality.
- Multiple turns contain the invocation string and the last intended occurrence is verified.
- Shared prefix length varies so cache reuse can be measured separately from quality.

## Performance measurement

Measure quality first. Then measure:

- Time to first token.
- End-to-end latency.
- Prompt and completion token counts.
- Throughput under a declared concurrency.
- GPU memory and cache behavior.
- Cold and warm runs separately.

Include hardware, software versions, batching, context lengths, concurrency, and sample counts. Compare LoRA and aLoRA on the same workload. Do not copy upstream demonstration numbers as guarantees for a different adapter or environment.

## Release evidence

Archive:

```text
eval-report.json
case-results.jsonl
environment.json
prompt-snapshots/
error-analysis.md
```

Each case result should include case ID, backend, model build hash, adapter technology, raw completion, parsed result, expected result, pass or fail reason, latency, and token counts. Exclude sensitive raw inputs from broad logs.

Next: [09 - Troubleshooting](09-troubleshooting.md).
