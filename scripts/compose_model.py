#!/usr/bin/env python3
"""Generate or execute the Granite Switch Composer command for a custom adapter."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="ibm-granite/granite-4.1-3b")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./composed-model"))
    parser.add_argument("--with-library", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        "python",
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        args.base_model,
        "--adapters",
        str(args.adapter),
        *args.with_library,
        "--output",
        str(args.output),
    ]


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print(shlex.join(command))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
