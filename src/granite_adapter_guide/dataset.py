"""Validation for the guide's JSONL training data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetValidationReport:
    """Result of validating one or two dataset splits."""

    ok: bool = True
    record_count: int = 0
    files: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _id_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def validate_dataset(
    train_path: str | Path,
    validation_path: str | Path | None = None,
) -> DatasetValidationReport:
    """Validate train and optional validation JSONL files.

    Every non-blank line must be a JSON object containing string ``input`` and
    ``output`` fields. When records contain ``id``, duplicates are rejected
    across both splits as well as within a split.
    """

    report = DatasetValidationReport()
    seen_ids: dict[str, tuple[Any, Path, int]] = {}
    paths = [Path(train_path)]
    if validation_path is not None:
        paths.append(Path(validation_path))

    for path in paths:
        count = 0
        if not path.is_file():
            report.errors.append(f"{path}: file does not exist")
            report.files[str(path)] = 0
            continue

        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                count += 1
                location = f"{path}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    report.errors.append(f"{location}: invalid JSON: {exc.msg}")
                    continue
                if not isinstance(record, dict):
                    report.errors.append(f"{location}: record must be a JSON object")
                    continue

                for field_name in ("input", "output"):
                    if field_name not in record:
                        report.errors.append(f"{location}: missing required field '{field_name}'")
                    elif not isinstance(record[field_name], str):
                        report.errors.append(f"{location}: {field_name} must be a string")
                    elif not record[field_name].strip():
                        report.errors.append(f"{location}: {field_name} must not be empty")

                if "id" in record:
                    key = _id_key(record["id"])
                    if key in seen_ids:
                        value, first_path, first_line = seen_ids[key]
                        report.errors.append(
                            f"{location}: duplicate id {value!r}; first seen at "
                            f"{first_path}:{first_line}"
                        )
                    else:
                        seen_ids[key] = (record["id"], path, line_number)

        report.files[str(path)] = count
        report.record_count += count
        if count == 0:
            report.errors.append(f"{path}: contains no records")

    report.ok = not report.errors
    return report
