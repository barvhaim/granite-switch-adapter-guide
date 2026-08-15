# 10 - Production Checklist

Use this as a release gate. Every checked item should point to evidence, not memory.

## Contract and risk

- [ ] The adapter implements one narrow, versioned function.
- [ ] Inputs, outputs, JSON Schema, edge cases, and error policy are approved.
- [ ] The application validates inputs before invocation.
- [ ] The application rejects invalid structured outputs safely.
- [ ] High-impact false positives and false negatives have named owners and mitigations.
- [ ] Human review or fallback behavior exists where the adapter is not authorized to decide alone.

## Data governance

- [ ] Training, validation, and test JSONL parse cleanly.
- [ ] Targets validate against the production output schema.
- [ ] Duplicates and leakage across splits were checked.
- [ ] Provenance, license, consent, retention, and deletion requirements are recorded.
- [ ] Secrets, proprietary data, and personal information were removed or explicitly approved.
- [ ] The test set is frozen and hash-addressed.
- [ ] Annotation guidelines and known disagreement areas are archived.

## Training reproducibility

- [ ] Base model is `ibm-granite/granite-4.1-3b` or the approved replacement, with an exact revision.
- [ ] Tokenizer and chat template identity are pinned.
- [ ] Training code, dependencies, hardware class, seeds, and hyperparameters are recorded.
- [ ] Response-only masking was inspected on tokenized examples.
- [ ] LoRA was trained and evaluated before aLoRA.
- [ ] PEFT output contains the approved config and safetensors.
- [ ] No full base-model weights or optimizer states leaked into the release adapter package.

## aLoRA invariant

- [ ] `alora_invocation_tokens` is non-empty.
- [ ] The IDs equal the pinned tokenizer encoding of the approved invocation string.
- [ ] The invocation appears at the intended boundary in every training and evaluation prompt.
- [ ] Truncation cannot remove or split it.
- [ ] User-controlled text cannot create a later unintended activation occurrence.
- [ ] Direct PEFT and composed prompt snapshots show the expected activation boundary.
- [ ] Runtime requests use the composed chat template rather than manual control-token insertion.

## Package and composition

- [ ] Layout is `<adapter_name>/granite-4.1-3b/{lora|alora}/`.
- [ ] `adapter_config.json`, `adapter_model.safetensors`, and `io.yaml` are present.
- [ ] Directory name, `io.yaml` name, invocation name, and application configuration agree.
- [ ] `io.yaml` schema and generation parameters are reviewed.
- [ ] Granite Switch is used to compose, not train, the adapters.
- [ ] Composer version or source revision is pinned.
- [ ] `--list-adapters` shows only the intended names and technologies.
- [ ] `compose_report.json` has no unexplained missing, zero, or unexpected modules.
- [ ] `config.json`, `adapter_index.json`, copied `io_configs`, and `BUILD.md` were inspected.
- [ ] Complete composed directory hashes are archived.

## Quality and parity

- [ ] Direct PEFT LoRA passed the task-quality gate.
- [ ] Composed LoRA matches the approved reference within declared tolerances.
- [ ] Direct and composed aLoRA passed the same frozen test set.
- [ ] Hugging Face and vLLM parsed outputs meet backend parity criteria.
- [ ] JSON parse rate, schema validity, task metrics, and class-specific errors are reported.
- [ ] Known failures were reviewed, categorized, and either fixed or accepted by an owner.
- [ ] Performance was measured only after correctness, on declared hardware and workload.
- [ ] No benchmark number was borrowed from a different adapter or environment.

## Serving

- [ ] Hugging Face calls pass `adapter_name` to `apply_chat_template`.
- [ ] vLLM calls pass `adapter_name` through `chat_template_kwargs`.
- [ ] Unknown adapter names are rejected before inference.
- [ ] Temperature, token limits, stop behavior, timeout, and retry policy are explicit.
- [ ] Health checks exercise model readiness, not only process liveness.
- [ ] Capacity, concurrency, context-length limits, and GPU memory headroom are tested.
- [ ] Authentication, authorization, network policy, and rate limits are enabled as required.
- [ ] Mellea integration, if used, is pinned and tested for the exact backend path rather than assumed from general documentation.

## Observability and privacy

- [ ] Logs include request ID, model build ID, adapter name, backend, latency, token counts, parse status, and error category.
- [ ] Logs do not expose raw sensitive prompts or completions by default.
- [ ] Metrics alert on schema failures, empty outputs, latency, saturation, retries, and class-distribution drift.
- [ ] A sampled, privacy-reviewed failure-analysis workflow exists.
- [ ] Model and adapter build metadata are available during incident response.

## Security and supply chain

- [ ] Adapter, base-model, package, and container sources are allowlisted and pinned.
- [ ] Artifact hashes and signatures are verified where available.
- [ ] Safetensors is preferred over pickle-based weight formats.
- [ ] Dependency and container vulnerability scans meet policy.
- [ ] Public release data and model-card disclosures were reviewed.
- [ ] Malicious or malformed prompts cannot choose arbitrary local paths or adapter artifacts.

## Rollout and rollback

- [ ] Staging uses the same composed artifact promoted to production.
- [ ] Canary traffic and release success criteria are defined.
- [ ] Shadow or replay evaluation respects privacy and retention rules.
- [ ] Rollback restores the previous model build, tokenizer, template, and application schema together.
- [ ] Old artifacts remain available for the approved rollback window.
- [ ] On-call runbooks cover activation failures, invalid JSON, parity drift, OOM, and server startup.
- [ ] The release owner has signed the evidence bundle.

## Evidence bundle

Archive at least:

```text
contract.md
data-manifest.json
training-run.json
adapter-manifest.json
adapter-file-hashes.txt
compose_report.json
BUILD.md
eval-report.json
case-results.jsonl
environment.json
composed-file-hashes.txt
approval-record.md
rollback-record.json
```

The checklist is complete only when each required item has a link to one of these artifacts or an approved exception.

Sources used throughout the guide are collected in [SOURCES.md](SOURCES.md).
