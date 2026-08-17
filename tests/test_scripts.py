import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_script_help_is_dependency_light(script):
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_composed_runner_appends_visible_alora_invocation():
    script = ROOT / "scripts" / "run_composed_model.py"
    spec = importlib.util.spec_from_file_location("run_composed_model", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = argparse.Namespace(prompt="Contact alex@example.com", invocation="<pii_detection>")
    assert module.user_content(args) == "Contact alex@example.com\n<pii_detection>"

    args.invocation = None
    assert module.user_content(args) == "Contact alex@example.com"


def test_compose_script_prints_reproducible_command():
    adapter = "outputs/pii-library/pii_detection/granite-4.1-3b/alora"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/compose_model.py",
            "--adapter",
            adapter,
            "--with-library",
            "ibm-granite/granitelib-rag-r1.0",
            "--output",
            "composed-model",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "granite_switch.composer.compose_granite_switch" in result.stdout
    assert adapter in result.stdout
    assert "ibm-granite/granitelib-rag-r1.0" in result.stdout


def test_peft_runner_loads_and_writes_jsonl_batches(tmp_path):
    script = ROOT / "scripts" / "run_peft_adapter.py"
    spec = importlib.util.spec_from_file_location("run_peft_adapter", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    input_path = tmp_path / "test.jsonl"
    input_path.write_text(
        json.dumps({"id": "test-001", "input": "Classify this"}) + "\n",
        encoding="utf-8",
    )
    rows = module.load_batch_inputs(input_path)
    assert rows == [{"id": "test-001", "input": "Classify this"}]

    output_path = tmp_path / "predictions.jsonl"
    module.write_predictions(output_path, [{"id": "test-001", "output": '{"label":"ok"}'}])
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "id": "test-001",
        "output": '{"label":"ok"}',
    }


def test_peft_runner_rejects_invalid_batch_rows(tmp_path):
    script = ROOT / "scripts" / "run_peft_adapter.py"
    spec = importlib.util.spec_from_file_location("run_peft_adapter_invalid", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text(json.dumps({"id": "test-001"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input"):
        module.load_batch_inputs(input_path)
