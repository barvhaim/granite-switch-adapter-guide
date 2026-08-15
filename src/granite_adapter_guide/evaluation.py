"""Deterministic evaluation of JSON predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationReport:
    """Aggregate JSON evaluation counts."""

    expected_count: int = 0
    prediction_count: int = 0
    json_valid_count: int = 0
    schema_valid_count: int = 0
    exact_match_count: int = 0
    missing_count: int = 0
    exact_match_rate: float = 0.0
    json_valid_rate: float = 0.0
    schema_valid_rate: float = 0.0
    missing_ids: list[Any] = field(default_factory=list)
    unexpected_ids: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        return records, [f"{path}: {exc}"]
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(value, dict):
                errors.append(f"{path}:{line_number}: record must be a JSON object")
                continue
            records.append(value)
    return records, errors


def _output_text(record: dict[str, Any], prediction: bool = False) -> Any:
    if prediction and "prediction" in record:
        return record["prediction"]
    return record.get("output")


def evaluate_predictions(
    expected_path: str | Path,
    predictions_path: str | Path,
    schema_path: str | Path | None = None,
) -> EvaluationReport:
    """Compare separate expected and prediction JSONL files.

    Expected records use ``output``; prediction records use either ``output``
    or ``prediction``. Exact matching compares parsed JSON values, so
    insignificant whitespace and object key ordering do not affect the score.
    Missing predictions remain in the expected-record denominator. When a
    schema is supplied, validation uses the installed ``jsonschema`` package.
    """

    expected, expected_errors = _read_jsonl(Path(expected_path))
    predictions, prediction_errors = _read_jsonl(Path(predictions_path))
    report = EvaluationReport(
        expected_count=len(expected),
        prediction_count=len(predictions),
        errors=expected_errors + prediction_errors,
    )

    validator: Any | None = None
    if schema_path is not None:
        try:
            loaded_schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            if not isinstance(loaded_schema, dict):
                raise ValueError("schema must be a JSON object")
            from jsonschema import Draft202012Validator

            Draft202012Validator.check_schema(loaded_schema)
            validator = Draft202012Validator(loaded_schema)
        except ImportError:
            report.errors.append("schema validation requires the optional 'jsonschema' package")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            report.errors.append(f"{schema_path}: invalid schema: {exc}")
        except Exception as exc:  # jsonschema.SchemaError without a hard dependency
            report.errors.append(f"{schema_path}: invalid schema: {exc}")

    use_ids = bool(expected) and all("id" in record for record in expected)
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    if use_ids:
        prediction_by_id: dict[str, dict[str, Any]] = {}
        raw_ids: dict[str, Any] = {}
        for prediction in predictions:
            if "id" not in prediction:
                report.errors.append("prediction record is missing id")
                continue
            key = json.dumps(prediction["id"], sort_keys=True, ensure_ascii=False)
            if key in prediction_by_id:
                report.errors.append(f"duplicate prediction id {prediction['id']!r}")
                continue
            prediction_by_id[key] = prediction
            raw_ids[key] = prediction["id"]
        expected_keys: set[str] = set()
        for expected_record in expected:
            key = json.dumps(expected_record["id"], sort_keys=True, ensure_ascii=False)
            expected_keys.add(key)
            prediction = prediction_by_id.get(key)
            pairs.append((expected_record, prediction))
            if prediction is None:
                report.missing_ids.append(expected_record["id"])
        report.unexpected_ids = sorted(
            (raw_ids[key] for key in prediction_by_id.keys() - expected_keys), key=str
        )
    else:
        pairs = [
            (expected_record, predictions[index] if index < len(predictions) else None)
            for index, expected_record in enumerate(expected)
        ]
        if len(predictions) > len(expected):
            report.unexpected_ids = list(range(len(expected) + 1, len(predictions) + 1))

    report.missing_count = sum(prediction is None for _, prediction in pairs)
    for expected_record, prediction in pairs:
        if prediction is None:
            continue
        prediction_text = _output_text(prediction, prediction=True)
        if not isinstance(prediction_text, str):
            report.errors.append("prediction output must be a string")
            continue
        try:
            prediction_value = json.loads(prediction_text)
        except json.JSONDecodeError:
            continue
        report.json_valid_count += 1
        if schema_path is None or (validator is not None and validator.is_valid(prediction_value)):
            report.schema_valid_count += 1

        expected_text = _output_text(expected_record)
        if not isinstance(expected_text, str):
            report.errors.append("expected output must be a string")
            continue
        try:
            expected_value = json.loads(expected_text)
        except json.JSONDecodeError:
            report.errors.append("expected output is not valid JSON")
            continue
        if prediction_value == expected_value:
            report.exact_match_count += 1

    denominator = report.expected_count
    if denominator:
        report.exact_match_rate = report.exact_match_count / denominator
        report.json_valid_rate = report.json_valid_count / denominator
        report.schema_valid_rate = report.schema_valid_count / denominator
    return report
