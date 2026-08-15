"""Command-line interface for the lightweight guide helpers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from .adapter import validate_adapter
from .dataset import validate_dataset
from .evaluation import evaluate_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="granite-adapter-guide")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("validate-dataset", help="validate training JSONL")
    dataset.add_argument("train", help="training JSONL path")
    dataset.add_argument(
        "validation", nargs="?", default=None, help="optional validation JSONL path"
    )

    adapter = subparsers.add_parser("validate-adapter", help="validate an adapter directory")
    adapter.add_argument("adapter", help="lora or alora technology directory")

    evaluate = subparsers.add_parser("evaluate", help="evaluate JSONL predictions")
    evaluate.add_argument("expected", help="expected dataset JSONL")
    evaluate.add_argument("predictions", help="prediction JSONL")
    evaluate.add_argument("--schema", default=None, help="optional JSON Schema file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-dataset":
        report = validate_dataset(args.train, args.validation)
        exit_code = 0 if report.ok else 1
    elif args.command == "validate-adapter":
        report = validate_adapter(args.adapter)
        exit_code = 0 if report.ok else 1
    else:
        report = evaluate_predictions(args.expected, args.predictions, args.schema)
        exit_code = 0 if not report.errors else 1

    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return exit_code
