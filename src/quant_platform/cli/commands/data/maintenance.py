"""Data maintenance command registration."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from quant_platform.application.data import MaintainDataRequest
from quant_platform.cli.registry import bind_command


def register_maintenance(sub: Any) -> None:
    maint_p = sub.add_parser(
        "maintain",
        help="Run the DataMaintenanceSupervisor loop (bar ingest + feature jobs).",
    )
    maint_p.add_argument(
        "--interval",
        type=float,
        default=900.0,
        help="Seconds between maintenance ticks. Default: 900 (15 minutes).",
    )
    maint_p.add_argument(
        "--backfill-start",
        type=date.fromisoformat,
        default=None,
        help=(
            "Inclusive start date (YYYY-MM-DD) for a one-shot backfill. "
            "When supplied together with --backfill-end, the supervisor "
            "runs backfill_once and exits instead of entering the loop."
        ),
    )
    maint_p.add_argument("--backfill-end", type=date.fromisoformat, default=None)
    maint_p.add_argument(
        "--contracts-file",
        default=None,
        help=(
            "Optional JSON mapping instrument_id UUID -> contract spec. "
            "Required for --backfill-* against the IB gateway."
        ),
    )
    bind_command(
        maint_p,
        use_case_name="data.maintain",
        request_factory=lambda args: MaintainDataRequest(
            interval_seconds=args.interval,
            backfill_start=args.backfill_start,
            backfill_end=args.backfill_end,
            contracts_file=Path(args.contracts_file) if args.contracts_file else None,
        ),
        request_type=MaintainDataRequest,
    )


__all__ = ["register_maintenance"]
