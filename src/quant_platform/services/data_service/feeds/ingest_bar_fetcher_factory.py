"""Build the :class:`~quant_platform.services.data_service.ingest.daily_ingest.BarFetcher`
for CLI ingest and maintenance when IB is primary and one or more secondary
vendors are optionally enabled (Phase 4 / R-DAT-04).

Vendor resolution order:
1. ``bar_fetch_fallback_chain`` (list) — when non-empty, builds a
   ``FailoverBarFetcher`` with all listed vendors as secondaries (IB primary).
2. ``bar_fetch_fallback`` (single string) — backward-compat single-vendor path.
3. No fallback — returns the bare ``IBBarFetcher``.

Supported vendor names: ``"tiingo"``, ``"polygon"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from quant_platform.services.data_service.feeds.failover_bar_fetcher import (
    BarFetcher as FetcherAlias,
)
from quant_platform.services.data_service.feeds.failover_bar_fetcher import FailoverBarFetcher
from quant_platform.services.data_service.feeds.ib_bar_fetcher import IBBarFetcher
from quant_platform.services.data_service.feeds.polygon_daily_bar_fetcher import (
    PolygonDailyBarFetcher,
)
from quant_platform.services.data_service.feeds.tiingo_bar_fetcher import TiingoBarFetcher

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings
    from quant_platform.services.data_service.ingest.daily_ingest import BarFetcher

log = structlog.get_logger(__name__)


def _build_secondary(name: str, d: object, bar_seconds: int) -> FetcherAlias | None:
    """Construct a single secondary fetcher by vendor name, or None if unconfigured."""
    if name == "tiingo":
        token = getattr(d, "tiingo_api_token", "")
        if not (isinstance(token, str) and token.strip()):
            return None
        return TiingoBarFetcher(token, bar_seconds=bar_seconds)
    if name == "polygon":
        key = getattr(d, "polygon_api_key", "")
        if not (isinstance(key, str) and key.strip()):
            return None
        return PolygonDailyBarFetcher(
            key,
            bar_seconds=bar_seconds,
            base_url=getattr(d, "polygon_base_url", "https://api.polygon.io"),
            max_concurrent=getattr(d, "polygon_max_concurrent", 4),
            timeout_seconds=getattr(d, "polygon_timeout_seconds", 30.0),
        )
    return None


def build_ingest_bar_fetcher(
    settings: PlatformSettings,
    broker: object,
    *,
    bar_seconds: int = 86400,
) -> BarFetcher | None:
    """Return a fetcher, or None when the broker has no historical API."""
    if not hasattr(broker, "fetch_historical_bars"):
        return None
    primary: BarFetcher = IBBarFetcher(broker, bar_seconds=bar_seconds)
    d = settings.data_ingest

    # Multi-vendor chain supersedes the single-fallback field.
    vendor_names: list[str] = list(d.bar_fetch_fallback_chain) or (
        [d.bar_fetch_fallback] if d.bar_fetch_fallback != "none" else []
    )
    if not vendor_names:
        return primary

    secondaries: list[FetcherAlias] = []
    names_used: list[str] = []
    for name in vendor_names:
        fetcher = _build_secondary(name, d, bar_seconds)
        if fetcher is not None:
            secondaries.append(fetcher)
            names_used.append(name)
        else:
            log.warning("ingest_bar_fetcher.secondary_skipped", vendor=name, reason="unconfigured")

    if not secondaries:
        return primary

    log.info("ingest_bar_fetcher.chain", secondaries=names_used)
    return FailoverBarFetcher(
        primary,
        secondaries=secondaries,
        primary_name="ib",
        secondary_names=names_used,
    )


def build_vendor_bar_fetcher(
    settings: PlatformSettings,
    *,
    bar_seconds: int = 86400,
) -> BarFetcher | None:
    """Return a vendor-only fetcher (no IB) for large historical backfills.

    Builds the configured Tiingo/Polygon vendors from ``bar_fetch_fallback_chain``
    (or the single ``bar_fetch_fallback``).  The first configured vendor is the
    primary; any others become failover secondaries.  Returns ``None`` when no
    vendor is configured.
    """
    d = settings.data_ingest
    vendor_names: list[str] = list(d.bar_fetch_fallback_chain) or (
        [d.bar_fetch_fallback] if d.bar_fetch_fallback != "none" else []
    )

    fetchers: list[FetcherAlias] = []
    names_used: list[str] = []
    for name in vendor_names:
        fetcher = _build_secondary(name, d, bar_seconds)
        if fetcher is not None:
            fetchers.append(fetcher)
            names_used.append(name)
        else:
            log.warning("vendor_bar_fetcher.skipped", vendor=name, reason="unconfigured")

    if not fetchers:
        return None
    if len(fetchers) == 1:
        log.info("vendor_bar_fetcher.single", vendor=names_used[0])
        return fetchers[0]

    log.info("vendor_bar_fetcher.chain", vendors=names_used)
    return FailoverBarFetcher(
        fetchers[0],
        secondaries=fetchers[1:],
        primary_name=names_used[0],
        secondary_names=names_used[1:],
    )
