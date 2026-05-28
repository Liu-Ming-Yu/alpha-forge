"""CLI application bootstrap."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.cli.commands import all_command_specs
from quant_platform.cli.context import create_context
from quant_platform.cli.registry import build_parser as build_registered_parser
from quant_platform.cli.registry import dispatch

if TYPE_CHECKING:
    import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the production CLI parser."""
    return build_registered_parser(all_command_specs())


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    context = create_context()
    return dispatch(args, context)
