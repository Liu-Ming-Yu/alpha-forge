"""Shared CLI runtime context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.cli.cli_application import build_cli_use_cases
from quant_platform.config import PlatformSettings, configure_logging

if TYPE_CHECKING:
    from quant_platform.application.results import UseCaseResult
    from quant_platform.application.use_cases import UseCaseRegistry


@dataclass(frozen=True)
class CLIContext:
    """Dependencies shared by command handlers after argument parsing."""

    use_cases: UseCaseRegistry

    async def run(self, use_case_name: str, request: object) -> UseCaseResult[object]:
        """Execute one named application use case."""
        return await self.use_cases.run(use_case_name, request)


def create_context() -> CLIContext:
    """Build the CLI context after argparse has accepted the command."""
    settings = PlatformSettings()
    configure_logging(settings.logging)
    return CLIContext(use_cases=build_cli_use_cases(settings))
