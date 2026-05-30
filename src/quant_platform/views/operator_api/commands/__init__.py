"""Generic CLI-command exposure for the operator console.

Introspects the argparse CLI into a JSON catalog and runs commands as tracked
subprocess jobs, so every ``python -m quant_platform`` command is available
from the browser console without hand-coding a screen per command.
"""

from __future__ import annotations

from quant_platform.views.operator_api.commands.catalog import (
    build_command_catalog,
    find_command,
    reconstruct_argv,
)
from quant_platform.views.operator_api.commands.jobs import JobStore, get_job_store

__all__ = [
    "JobStore",
    "build_command_catalog",
    "find_command",
    "get_job_store",
    "reconstruct_argv",
]
