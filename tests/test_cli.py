import json
import os
import subprocess
import sys


def run_cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(__import__("pathlib").Path(__file__).parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-m", "granite_adapter_guide", *map(str, args)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_validate_dataset_cli_returns_json_and_status(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"input": "x", "output": "{}"}) + "\n")
    result = run_cli("validate-dataset", dataset)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True

    dataset.write_text("bad\n")
    result = run_cli("validate-dataset", dataset)
    assert result.returncode == 1
    assert json.loads(result.stdout)["ok"] is False


def test_cli_help_lists_all_commands():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "validate-dataset" in result.stdout
    assert "validate-adapter" in result.stdout
    assert "evaluate" in result.stdout
