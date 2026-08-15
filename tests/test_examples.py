import json
import shutil
from pathlib import Path

from granite_adapter_guide.adapter import validate_adapter
from granite_adapter_guide.dataset import validate_dataset
from granite_adapter_guide.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pii_detection"


def test_included_dataset_and_predictions_are_valid():
    dataset_report = validate_dataset(EXAMPLE / "train.jsonl", EXAMPLE / "validation.jsonl")
    assert dataset_report.ok, dataset_report.errors
    assert dataset_report.record_count == 9

    evaluation_report = evaluate_predictions(
        EXAMPLE / "validation.jsonl",
        EXAMPLE / "predictions.example.jsonl",
        schema_path=EXAMPLE / "schema.json",
    )
    assert evaluation_report.errors == []
    assert evaluation_report.expected_count == 3
    assert evaluation_report.exact_match_count == 3
    assert evaluation_report.schema_valid_count == 3


def test_included_io_contract_forms_a_valid_alora_layout(tmp_path):
    adapter = tmp_path / "pii_detection" / "granite-4.1-3b" / "alora"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"example-test-placeholder")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "alora_invocation_tokens": [27, 123, 29],
            }
        ),
        encoding="utf-8",
    )
    shutil.copy(EXAMPLE / "io.yaml", adapter / "io.yaml")

    report = validate_adapter(adapter)
    assert report.ok, report.errors
