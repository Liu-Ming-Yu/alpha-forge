"""Data and event-bus command registrations."""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path
from typing import Any

from quant_platform.application.data import (
    ComputeFeaturesRequest,
    IngestRequest,
    ReprocessCorporateActionsRequest,
)
from quant_platform.cli.commands.data.event_bus import register_event_bus
from quant_platform.cli.commands.data.intraday import register_intraday
from quant_platform.cli.commands.data.maintenance import register_maintenance
from quant_platform.cli.registry import bind_command


def register(sub: Any) -> None:
    feat_p = sub.add_parser(
        "compute-features",
        help="Run the feature pipeline (bar data -> FeatureRepository).",
    )
    feat_p.add_argument(
        "--contracts-file",
        default=None,
        help=(
            "Optional path to JSON mapping instrument_id UUID -> metadata. "
            "When provided, liquidity and feature jobs run for this universe."
        ),
    )
    bind_command(
        feat_p,
        use_case_name="data.compute_features",
        request_factory=lambda args: ComputeFeaturesRequest(
            contracts_file=Path(args.contracts_file) if args.contracts_file else None,
        ),
        request_type=ComputeFeaturesRequest,
    )

    ing_p = sub.add_parser(
        "ingest",
        help=(
            "Backfill the Parquet bar store over a date window from the "
            "configured broker (typically IB Gateway)."
        ),
    )
    ing_p.add_argument("--start", required=True, type=date.fromisoformat)
    ing_p.add_argument("--end", required=True, type=date.fromisoformat)
    ing_p.add_argument(
        "--contracts-file",
        required=True,
        help=(
            "Path to JSON mapping instrument_id UUID -> IB contract spec "
            "(same schema as run-engine --contracts-file)."
        ),
    )
    ing_p.add_argument(
        "--bar-seconds",
        type=int,
        default=86400,
        help="Bar granularity in seconds. Default: 86400 (daily).",
    )
    ing_p.add_argument(
        "--data-source",
        choices=("ib", "vendor"),
        default="ib",
        help=(
            "Bar source. 'ib' uses the broker historical API (default). "
            "'vendor' uses the configured Tiingo/Polygon feed and skips IB "
            "entirely — required for large historical backfills where IB "
            "pacing limits make per-instrument fetches infeasible."
        ),
    )
    bind_command(
        ing_p,
        use_case_name="data.ingest",
        request_factory=lambda args: IngestRequest(
            start=args.start,
            end=args.end,
            contracts_file=Path(args.contracts_file),
            bar_seconds=args.bar_seconds,
            source=args.data_source,
        ),
        request_type=IngestRequest,
    )

    ca_p = sub.add_parser(
        "reprocess-ca",
        help=(
            "Re-emit adjusted bar partitions for an instrument after a "
            "late corporate-action correction (writes to bars_adjusted/)."
        ),
    )
    ca_p.add_argument(
        "--instrument-id",
        required=True,
        type=uuid.UUID,
        help="Instrument UUID whose bars should be re-emitted.",
    )
    bind_command(
        ca_p,
        use_case_name="data.reprocess_ca",
        request_factory=lambda args: ReprocessCorporateActionsRequest(
            instrument_id=args.instrument_id,
        ),
        request_type=ReprocessCorporateActionsRequest,
    )

    register_intraday(sub)


__all__ = ["register", "register_event_bus", "register_maintenance"]
