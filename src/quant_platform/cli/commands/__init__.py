"""CLI command group registration."""

from __future__ import annotations

from quant_platform.cli.commands import (
    broker,
    data,
    engines,
    governance,
    migrations,
    research,
    runtime,
    text_events,
)
from quant_platform.cli.registry import CommandSpec


def all_command_specs() -> tuple[CommandSpec, ...]:
    """Return top-level command groups in public help order."""
    return (
        CommandSpec("runtime", runtime.register),
        CommandSpec("broker", broker.register),
        CommandSpec("engines", engines.register),
        CommandSpec("data", data.register),
        CommandSpec("migrations", migrations.register),
        CommandSpec("maintenance", data.register_maintenance),
        CommandSpec("research", research.register),
        CommandSpec("api", runtime.register_api),
        CommandSpec("event-bus", data.register_event_bus),
        CommandSpec("text-events", text_events.register),
        CommandSpec("governance", governance.register),
        CommandSpec("smoke", runtime.register_smoke),
    )
