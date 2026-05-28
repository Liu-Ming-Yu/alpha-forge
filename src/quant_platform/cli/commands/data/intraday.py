"""Intraday data command registrations."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_platform.application.operator.requests import (
    IntradayFetchRequest,
    IntradayImportRequest,
    IntradayQuorumRequest,
    IntradayValidateRequest,
)
from quant_platform.cli.registry import bind_command

if TYPE_CHECKING:
    import argparse


def register_intraday(sub: Any) -> None:
    intraday_p = sub.add_parser(
        "intraday",
        help="Vendor-neutral intraday bar import, validation, and quorum utilities.",
    )
    intraday_sub = intraday_p.add_subparsers(dest="intraday_command", required=True)

    validate_p = intraday_sub.add_parser(
        "validate",
        help="Validate a vendor CSV/Parquet file against the canonical 1-minute OHLCV schema.",
    )
    validate_p.add_argument("--input", required=True, type=Path)
    validate_p.add_argument("--vendor", required=True)
    validate_p.add_argument("--contracts-file", required=True)
    validate_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    bind_command(
        validate_p,
        use_case_name="data.intraday",
        request_factory=_validate_request,
        request_type=IntradayValidateRequest,
    )

    import_p = intraday_sub.add_parser(
        "import",
        help="Validate and store a vendor CSV/Parquet file in the canonical Parquet bar store.",
    )
    import_p.add_argument("--input", required=True, type=Path)
    import_p.add_argument("--vendor", required=True)
    import_p.add_argument("--contracts-file", required=True)
    import_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    import_p.add_argument("--allow-quarantined", action="store_true")
    bind_command(
        import_p,
        use_case_name="data.intraday",
        request_factory=_import_request,
        request_type=IntradayImportRequest,
    )

    fetch_p = intraday_sub.add_parser(
        "fetch",
        help="Fetch 1-minute bars from a configured external vendor and store them.",
    )
    fetch_p.add_argument("--vendor", required=True, choices=["polygon"])
    fetch_p.add_argument("--contracts-file", required=True)
    fetch_p.add_argument("--start", required=True, type=datetime.fromisoformat)
    fetch_p.add_argument("--end", required=True, type=datetime.fromisoformat)
    fetch_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    fetch_p.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional immutable CSV/Parquet freeze file for research screens.",
    )
    fetch_p.add_argument("--allow-quarantined", action="store_true")
    bind_command(
        fetch_p,
        use_case_name="data.intraday",
        request_factory=_fetch_request,
        request_type=IntradayFetchRequest,
    )

    quorum_p = intraday_sub.add_parser(
        "quorum",
        help="Compute 1-minute multi-vendor quorum evidence from two or more files.",
    )
    quorum_p.add_argument(
        "--vendor-file",
        action="append",
        required=True,
        help="Vendor/file pair in the form vendor=/path/to/file.csv. Repeat at least twice.",
    )
    quorum_p.add_argument("--contracts-file", required=True)
    quorum_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    quorum_p.add_argument("--required-vendor-count", type=int, default=2)
    quorum_p.add_argument("--max-disagreement-bps", type=Decimal, default=Decimal("50"))
    bind_command(
        quorum_p,
        use_case_name="data.intraday",
        request_factory=_quorum_request,
        request_type=IntradayQuorumRequest,
    )


def _validate_request(args: argparse.Namespace) -> IntradayValidateRequest:
    return IntradayValidateRequest(
        input=args.input,
        vendor=args.vendor,
        contracts_file=args.contracts_file,
        as_of=args.as_of,
    )


def _import_request(args: argparse.Namespace) -> IntradayImportRequest:
    return IntradayImportRequest(
        input=args.input,
        vendor=args.vendor,
        contracts_file=args.contracts_file,
        as_of=args.as_of,
        allow_quarantined=args.allow_quarantined,
    )


def _fetch_request(args: argparse.Namespace) -> IntradayFetchRequest:
    return IntradayFetchRequest(
        vendor=args.vendor,
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        as_of=args.as_of,
        output_file=args.output_file,
        allow_quarantined=args.allow_quarantined,
    )


def _quorum_request(args: argparse.Namespace) -> IntradayQuorumRequest:
    return IntradayQuorumRequest(
        vendor_file=tuple(args.vendor_file),
        contracts_file=args.contracts_file,
        as_of=args.as_of,
        required_vendor_count=args.required_vendor_count,
        max_disagreement_bps=args.max_disagreement_bps,
    )


__all__ = ["register_intraday"]
