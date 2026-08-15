import json

from granite_adapter_guide.dataset import validate_dataset


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_validates_train_and_validation_and_detects_cross_split_ids(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    write_jsonl(train, [{"id": "one", "input": "Question", "output": '{"ok": true}'}])
    write_jsonl(validation, [{"id": "two", "input": "Other", "output": '{"ok": false}'}])

    report = validate_dataset(train, validation)

    assert report.ok
    assert report.record_count == 2
    assert report.errors == []

    write_jsonl(validation, [{"id": "one", "input": "Other", "output": "{}"}])
    report = validate_dataset(train, validation)
    assert not report.ok
    assert any("duplicate id 'one'" in error for error in report.errors)


def test_reports_bad_json_missing_fields_and_wrong_types(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"input": "ok", "output": 3}\nnot json\n{"input": "missing"}\n')

    report = validate_dataset(train)

    assert not report.ok
    assert report.record_count == 3
    assert any("output must be a string" in error for error in report.errors)
    assert any("invalid JSON" in error for error in report.errors)
    assert any("missing required field 'output'" in error for error in report.errors)


def test_rejects_blank_input_or_output(tmp_path):
    train = tmp_path / "train.jsonl"
    write_jsonl(
        train,
        [
            {"id": "blank-input", "input": "   ", "output": "{}"},
            {"id": "blank-output", "input": "text", "output": ""},
        ],
    )

    report = validate_dataset(train)

    assert not report.ok
    assert any("input must not be empty" in error for error in report.errors)
    assert any("output must not be empty" in error for error in report.errors)


def test_empty_file_is_invalid(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("")
    report = validate_dataset(train)
    assert not report.ok
    assert any("contains no records" in error for error in report.errors)
