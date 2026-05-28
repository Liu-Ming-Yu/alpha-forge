"""Small argparse-based command registry."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quant_platform.application.errors import OperatorUsageError
from quant_platform.cli.presentation import render_result

if TYPE_CHECKING:
    from quant_platform.application.results import UseCaseResult
    from quant_platform.cli.context import CLIContext

CommandRequestFactory = Callable[[argparse.Namespace], object]


@dataclass(frozen=True)
class CommandSpec:
    """A top-level command group that can register parsers and handlers."""

    name: str
    register: Callable[[Any], None]


@dataclass(frozen=True)
class BoundCommand:
    """Application command metadata attached to an argparse parser."""

    use_case_name: str
    request_factory: CommandRequestFactory
    request_type: type[object]


def bind_command(
    parser: argparse.ArgumentParser,
    *,
    use_case_name: str,
    request_factory: CommandRequestFactory,
    request_type: type[object],
) -> None:
    """Attach typed application-command metadata to an argparse parser."""
    parser.set_defaults(
        _command=BoundCommand(
            use_case_name=use_case_name,
            request_factory=request_factory,
            request_type=request_type,
        )
    )


def build_parser(command_specs: Sequence[CommandSpec]) -> argparse.ArgumentParser:
    """Create the root parser and let command specs register themselves."""
    parser = argparse.ArgumentParser(
        prog="python -m quant_platform",
        description="Quant platform CLI - run strategy cycles, supervise, or check health.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for spec in command_specs:
        spec.register(sub)
    return parser


def dispatch(args: argparse.Namespace, context: CLIContext) -> int:
    """Run the selected application command and render its result."""
    _configure_windows_event_loop_policy()
    try:
        command = getattr(args, "_command", None)
        if command is None:
            raise RuntimeError(f"no CLI handler registered for command {args.command!r}")
        request = command.request_factory(args)
        return asyncio.run(_await_result(context.run(command.use_case_name, request)))
    except OperatorUsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


async def _await_result(result: Awaitable[UseCaseResult[Any]]) -> int:
    return render_result(await result)


def _configure_windows_event_loop_policy() -> None:
    # psycopg3 async requires SelectorEventLoop; Windows defaults to ProactorEventLoop.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


__all__ = [
    "BoundCommand",
    "CommandSpec",
    "bind_command",
    "build_parser",
    "dispatch",
]
