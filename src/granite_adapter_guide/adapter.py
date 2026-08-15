"""Validation for adapter directories consumed by Granite Switch."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AdapterValidationReport:
    """Result of checking a single adapter technology directory."""

    ok: bool = True
    path: str = ""
    technology: str | None = None
    errors: list[str] = field(default_factory=list)


def _load_json_object(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: invalid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path}: must contain a JSON object")
        return None
    return value


def validate_adapter(adapter_path: str | Path) -> AdapterValidationReport:
    """Validate a ``<adapter>/<model>/<lora|alora>`` adapter directory."""

    path = Path(adapter_path)
    technology = path.name if path.name in {"lora", "alora"} else None
    report = AdapterValidationReport(path=str(path), technology=technology)

    if not path.is_dir():
        report.errors.append(f"{path}: adapter directory does not exist")
        report.ok = False
        return report
    if technology is None:
        report.errors.append(f"{path}: technology directory must be named 'lora' or 'alora'")

    required = ("adapter_model.safetensors", "adapter_config.json", "io.yaml")
    for filename in required:
        if not (path / filename).is_file():
            report.errors.append(f"{path}: missing required file {filename}")

    config_path = path / "adapter_config.json"
    config = _load_json_object(config_path, report.errors) if config_path.is_file() else None
    if config is not None:
        if not isinstance(config.get("peft_type"), str) or not config["peft_type"].strip():
            report.errors.append(f"{config_path}: requires nonempty string 'peft_type'")
        rank = config.get("r")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            report.errors.append(f"{config_path}: requires positive integer 'r'")
        alpha = config.get("lora_alpha")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha <= 0:
            report.errors.append(f"{config_path}: requires positive numeric 'lora_alpha'")
        targets = config.get("target_modules")
        if (
            not isinstance(targets, list)
            or not targets
            or not all(isinstance(item, str) and item for item in targets)
        ):
            report.errors.append(f"{config_path}: requires nonempty string list 'target_modules'")
        if technology == "alora":
            tokens = config.get("alora_invocation_tokens")
            if (
                not isinstance(tokens, list)
                or not tokens
                or not all(
                    isinstance(token, int) and not isinstance(token, bool) and token >= 0
                    for token in tokens
                )
            ):
                report.errors.append(
                    f"{config_path}: alora_invocation_tokens must be a nonempty "
                    "list of nonnegative integers for alora"
                )

    io_path = path / "io.yaml"
    if io_path.is_file():
        try:
            io_config = yaml.safe_load(io_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            report.errors.append(f"{io_path}: invalid YAML: {exc}")
            io_config = None
        if not isinstance(io_config, dict):
            report.errors.append(f"{io_path}: must contain a YAML mapping")
        else:
            if not isinstance(io_config.get("name"), str) or not io_config["name"].strip():
                report.errors.append(f"{io_path}: requires nonempty 'name'")
            response_format = io_config.get("response_format")
            if response_format is not None:
                if isinstance(response_format, str):
                    try:
                        response_format = json.loads(response_format)
                    except json.JSONDecodeError as exc:
                        report.errors.append(
                            f"{io_path}: response_format must be valid JSON: {exc.msg}"
                        )
                        response_format = None
                if not isinstance(response_format, dict):
                    report.errors.append(
                        f"{io_path}: response_format must define a JSON schema object"
                    )

    report.ok = not report.errors
    return report
