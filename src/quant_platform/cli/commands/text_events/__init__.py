"""Text-event ingestion and extraction command registration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_platform.application.operator.requests import (
    ExtractTextFeaturesRequest,
    IngestNewsTextEventsRequest,
    IngestSecTextEventsRequest,
)
from quant_platform.cli.registry import bind_command

if TYPE_CHECKING:
    import argparse


def register(sub: Any) -> None:
    text_p = sub.add_parser(
        "text-events",
        help="Governed text-event ingestion and LLM feature extraction.",
    )
    text_sub = text_p.add_subparsers(dest="text_events_command", required=True)

    ingest_p = text_sub.add_parser("ingest-sec", help="Download and store SEC filings.")
    ingest_p.add_argument("--contracts-file", required=True)
    ingest_p.add_argument("--start", required=True, type=datetime.fromisoformat)
    ingest_p.add_argument("--end", required=True, type=datetime.fromisoformat)
    ingest_p.add_argument(
        "--cik-map-file",
        type=Path,
        default=Path("infra/config/sec_cik_map.json"),
    )
    ingest_p.add_argument("--sec-user-agent", default="")
    ingest_p.add_argument("--forms", nargs="+", default=["10-K", "10-Q", "8-K"])
    ingest_p.add_argument("--timeout-seconds", type=float, default=30.0)
    ingest_p.add_argument("--limit-per-symbol", type=int, default=None)
    ingest_p.add_argument(
        "--include-exhibits",
        action="store_true",
        help="Download preferred SEC exhibit documents in addition to primary filing documents.",
    )
    ingest_p.add_argument("--artifact-root", type=Path, default=None)
    bind_command(
        ingest_p,
        use_case_name="text_events",
        request_factory=_ingest_sec_request,
        request_type=IngestSecTextEventsRequest,
    )

    news_p = text_sub.add_parser("ingest-news", help="Download and store news headlines.")
    news_p.add_argument("--vendor", choices=["tws"], default="tws")
    news_p.add_argument("--contracts-file", required=True)
    news_p.add_argument("--start", required=True, type=datetime.fromisoformat)
    news_p.add_argument("--end", required=True, type=datetime.fromisoformat)
    news_p.add_argument(
        "--provider-codes",
        nargs="+",
        default=["BRFG", "BRFUPDN", "DJNL"],
        help="TWS API news provider codes. Values may also contain '+' or comma separators.",
    )
    news_p.add_argument(
        "--total-results-per-symbol",
        type=int,
        default=50,
        help="Maximum historical headlines to request per symbol from TWS (1-300).",
    )
    news_p.add_argument(
        "--headline-only",
        action="store_true",
        help="Skip reqNewsArticle calls and persist headline text only.",
    )
    news_p.add_argument("--artifact-root", type=Path, default=None)
    bind_command(
        news_p,
        use_case_name="text_events",
        request_factory=_ingest_news_request,
        request_type=IngestNewsTextEventsRequest,
    )

    extract_p = text_sub.add_parser(
        "extract-features",
        help="Extract event-level LLM text features into the feature repository.",
    )
    extract_p.add_argument("--start", required=True, type=datetime.fromisoformat)
    extract_p.add_argument("--end", required=True, type=datetime.fromisoformat)
    extract_p.add_argument(
        "--prompt-version",
        default="",
        help="Override QP__LLM__TEXT_PROMPT_VERSION for this extraction pass.",
    )
    extract_p.add_argument(
        "--document-role",
        choices=["all", "primary", "exhibit"],
        default="all",
        help="Restrict extraction to primary filing documents or exhibits.",
    )
    extract_p.add_argument(
        "--source-data-manifest",
        type=Path,
        default=None,
        help="Restrict extraction to events listed by a governed source-data manifest.",
    )
    extract_p.add_argument("--artifact-root", type=Path, default=None)
    extract_p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of LLM extraction calls to run in parallel (via "
            "asyncio.to_thread). 1 = sequential (default). Try 8 for backfills."
        ),
    )
    extract_p.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help=(
            "When set, write extraction progress as JSON every ~10s to this "
            "path. Use scripts/extract_status.py to read it."
        ),
    )
    extract_p.add_argument(
        "--per-call-timeout-seconds",
        type=float,
        default=180.0,
        help="Per-extraction hard timeout; stuck calls are marked failed.",
    )
    bind_command(
        extract_p,
        use_case_name="text_events",
        request_factory=_extract_features_request,
        request_type=ExtractTextFeaturesRequest,
    )


def _ingest_sec_request(args: argparse.Namespace) -> IngestSecTextEventsRequest:
    return IngestSecTextEventsRequest(
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        cik_map_file=args.cik_map_file,
        sec_user_agent=args.sec_user_agent,
        forms=tuple(args.forms),
        timeout_seconds=args.timeout_seconds,
        limit_per_symbol=args.limit_per_symbol,
        include_exhibits=args.include_exhibits,
        artifact_root=args.artifact_root,
    )


def _ingest_news_request(args: argparse.Namespace) -> IngestNewsTextEventsRequest:
    return IngestNewsTextEventsRequest(
        vendor=args.vendor,
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        provider_codes=tuple(args.provider_codes),
        total_results_per_symbol=args.total_results_per_symbol,
        include_article_text=not args.headline_only,
        artifact_root=args.artifact_root,
    )


def _extract_features_request(args: argparse.Namespace) -> ExtractTextFeaturesRequest:
    return ExtractTextFeaturesRequest(
        start=args.start,
        end=args.end,
        prompt_version=args.prompt_version,
        document_role=args.document_role,
        source_data_manifest=args.source_data_manifest,
        artifact_root=args.artifact_root,
        concurrency=args.concurrency,
        status_file=args.status_file,
        per_call_timeout_seconds=args.per_call_timeout_seconds,
    )
