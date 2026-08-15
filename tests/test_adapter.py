import json

from granite_adapter_guide.adapter import validate_adapter


def make_adapter(tmp_path, technology="lora", config=None, io_text=None):
    path = tmp_path / "adapter" / "granite-4.1-3b" / technology
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path / "adapter_config.json").write_text(
        json.dumps(
            config
            or {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
            }
        )
    )
    (path / "io.yaml").write_text(
        io_text
        or """name: demo
response_format: |
  {"type": "object", "required": ["answer"]}
"""
    )
    return path


def test_valid_lora_layout(tmp_path):
    report = validate_adapter(make_adapter(tmp_path))
    assert report.ok
    assert report.technology == "lora"
    assert report.errors == []


def test_io_response_format_is_optional(tmp_path):
    path = make_adapter(tmp_path, io_text="name: demo\n")
    assert validate_adapter(path).ok


def test_requires_lora_alpha_used_by_composer(tmp_path):
    path = make_adapter(
        tmp_path,
        config={"peft_type": "LORA", "r": 8, "target_modules": ["q_proj"]},
    )
    report = validate_adapter(path)
    assert not report.ok
    assert any("lora_alpha" in error for error in report.errors)


def test_alora_requires_nonempty_integer_invocation_tokens(tmp_path):
    path = make_adapter(tmp_path, "alora")
    report = validate_adapter(path)
    assert not report.ok
    assert any("alora_invocation_tokens" in error for error in report.errors)

    config = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj"],
        "alora_invocation_tokens": [100, 101],
    }
    path = make_adapter(tmp_path / "valid", "alora", config=config)
    assert validate_adapter(path).ok


def test_reports_layout_files_and_config_io_errors(tmp_path):
    wrong = tmp_path / "adapter"
    wrong.mkdir()
    report = validate_adapter(wrong)
    assert not report.ok
    assert any("technology directory" in error for error in report.errors)
    assert any("adapter_model.safetensors" in error for error in report.errors)

    path = make_adapter(tmp_path / "bad")
    (path / "adapter_config.json").write_text("[]")
    (path / "io.yaml").write_text("name: ''\nresponse_format: not-json\n")
    report = validate_adapter(path)
    assert not report.ok
    assert any("JSON object" in error for error in report.errors)
    assert any("nonempty 'name'" in error for error in report.errors)
    assert any("response_format" in error for error in report.errors)
