"""Governed news text-event ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from quant_platform.application.research.common import _load_instrument_contracts
from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema
from quant_platform.bootstrap.persistence.runtime_repositories import build_runtime_repositories
from quant_platform.bootstrap.text_events.manifests import event_manifest_records
from quant_platform.bootstrap.text_events.reports import instrument_coverage
from quant_platform.bootstrap.text_events.support import (
    blocked_text_payload,
    require_text_durable,
    text_artifact_root,
    text_event_slug,
    write_text_manifest,
)
from quant_platform.services.data_service.text.text_providers import (
    NewsArticleRecord,
    TWSNewsTextProvider,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.services.execution_service.ib.ib_news import IBNewsArticle

TWSNewsDownloadResult: TypeAlias = (
    tuple[list[NewsArticleRecord], dict[str, object]] | tuple[dict[str, object], dict[str, object]]
)


async def ingest_news_text_events(
    settings: PlatformSettings,
    *,
    vendor: str,
    contracts_file: str,
    start: datetime,
    end: datetime,
    provider_codes: tuple[str, ...],
    total_results_per_symbol: int,
    include_article_text: bool,
    artifact_root: Path | None,
) -> dict[str, object]:
    """Ingest vendor news into the durable text-event store."""
    require_text_durable(settings)
    await verify_postgres_schema(settings)
    normalized_vendor = vendor.strip().lower()
    slug = text_event_slug(f"news-{normalized_vendor}", start, end)
    if normalized_vendor != "tws":
        return blocked_text_payload(settings, slug, f"unsupported news vendor: {vendor}")
    if not 1 <= total_results_per_symbol <= 300:
        return blocked_text_payload(
            settings,
            slug,
            "total_results_per_symbol must be in [1, 300]",
        )

    contracts = _load_instrument_contracts(contracts_file)
    records, download_summary = await _download_tws_news_records(
        settings,
        contracts=contracts,
        start=start,
        end=end,
        provider_codes=_normalize_provider_codes(provider_codes),
        total_results_per_symbol=total_results_per_symbol,
        include_article_text=include_article_text,
        slug=slug,
    )
    if isinstance(records, dict):
        return records
    if not records:
        return blocked_text_payload(
            settings,
            slug,
            "TWS news ingestion returned zero records",
            download_summary,
        )

    repositories = build_runtime_repositories(settings)
    resolved_artifact_root = Path(artifact_root or text_artifact_root(settings) / "news")
    provider = TWSNewsTextProvider(records=records, artifact_root=resolved_artifact_root)
    events = await provider.ingest(repositories.text_event_store)
    manifest_path = write_text_manifest(
        settings,
        slug,
        "source_data_manifest.json",
        {
            "passed": True,
            "source": "news",
            "vendor": normalized_vendor,
            "download": download_summary,
            "events_ingested": len(events),
            "instrument_coverage": instrument_coverage(events),
            "events_by_symbol": _events_by_symbol(events),
            "events": event_manifest_records(events),
            "content_hashes": sorted(
                str(event.metadata.get("content_hash", "")) for event in events
            ),
            "artifact_root": str(resolved_artifact_root),
        },
    )
    return {
        "passed": True,
        "source_data_manifest": str(manifest_path),
        "events_ingested": len(events),
        "records_downloaded": len(records),
        "artifact_root": str(resolved_artifact_root),
    }


async def _download_tws_news_records(
    settings: PlatformSettings,
    *,
    contracts: dict[uuid.UUID, dict[str, object]],
    start: datetime,
    end: datetime,
    provider_codes: tuple[str, ...],
    total_results_per_symbol: int,
    include_article_text: bool,
    slug: str,
) -> TWSNewsDownloadResult:
    try:
        from quant_platform.core.exceptions import BrokerUnavailableError
        from quant_platform.services.execution_service.gateways.broker_gateway import (
            IBGatewayBrokerGateway,
        )
    except ImportError as exc:
        return blocked_text_payload(settings, slug, f"ibapi is required for TWS news: {exc}"), {}

    gateway = IBGatewayBrokerGateway(settings.broker, instrument_contracts=contracts)
    records: list[NewsArticleRecord] = []
    skipped_missing_con_id = _missing_con_id_count(contracts)
    try:
        await gateway.connect()
        for instrument_id in sorted(contracts, key=str):
            articles = await gateway.fetch_historical_news(
                instrument_id=instrument_id,
                start=start,
                end=end,
                provider_codes=provider_codes,
                total_results=total_results_per_symbol,
                include_article_text=include_article_text,
            )
            records.extend(_record_from_tws_article(article) for article in articles)
    except (BrokerUnavailableError, TimeoutError, OSError, RuntimeError) as exc:
        return blocked_text_payload(settings, slug, f"TWS news unavailable: {exc}"), {}
    finally:
        await gateway.disconnect()

    ready = sum(1 for record in records if record.article_text.strip())
    return records, {
        "vendor": "tws",
        "provider_codes": list(provider_codes),
        "instruments_requested": len(contracts),
        "contracts_missing_con_id": skipped_missing_con_id,
        "headlines_downloaded": len(records),
        "article_body_records": ready,
        "headline_only_records": len(records) - ready,
        "include_article_text": include_article_text,
    }


def _record_from_tws_article(article: IBNewsArticle) -> NewsArticleRecord:
    return NewsArticleRecord(
        vendor="tws",
        provider_code=article.provider_code,
        article_id=article.article_id,
        headline=article.headline,
        published_at=article.published_at,
        source_uri=f"ibkr://news/{article.provider_code}/{article.article_id}",
        article_text=article.article_text,
        article_type=article.article_type,
        instrument_id=article.instrument_id,
        symbol=article.symbol,
        metadata={
            "con_id": str(article.con_id),
            "tws_time_raw": article.raw_published_at,
            "article_status": article.article_status,
        },
    )


def _normalize_provider_codes(provider_codes: tuple[str, ...]) -> tuple[str, ...]:
    codes: list[str] = []
    for raw in provider_codes:
        for part in str(raw).replace(",", "+").split("+"):
            code = part.strip().upper()
            if code and code not in codes:
                codes.append(code)
    return tuple(codes or ("BRFG", "BRFUPDN", "DJNL"))


def _missing_con_id_count(contracts: dict[uuid.UUID, dict[str, object]]) -> int:
    return sum(1 for spec in contracts.values() if not isinstance(spec.get("con_id"), int))


def _events_by_symbol(events: Sequence[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        symbol = str(getattr(event, "metadata", {}).get("symbol", "")).upper().strip()
        if not symbol:
            continue
        counts[symbol] = counts.get(symbol, 0) + 1
    return dict(sorted(counts.items()))


__all__ = ["ingest_news_text_events"]
