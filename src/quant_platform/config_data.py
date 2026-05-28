"""Market-data ingest settings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DataIngestSettings(BaseModel):
    """Bar-ingest data sources (daily Parquet backfill, maintenance backfill)."""

    bar_fetch_fallback: Literal["none", "tiingo", "polygon"] = "none"
    bar_fetch_fallback_chain: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of secondary vendors for multi-vendor failover "
            '(e.g. ["tiingo", "polygon"]).  When non-empty, supersedes '
            '``bar_fetch_fallback``.  Allowed values: "tiingo", "polygon". '
            'Set via QP__DATA_INGEST__BAR_FETCH_FALLBACK_CHAIN=["tiingo","polygon"].'
        ),
    )
    tiingo_api_token: str = Field(
        default="",
        description=(
            "When ``bar_fetch_fallback=tiingo``, used as the Tiingo API token. "
            "Leave empty to disable the secondary even if ``tiingo`` is selected."
        ),
    )
    polygon_api_key: str = Field(
        default="",
        description=(
            "Polygon.io API key used by the intraday historical bar adapter. "
            "Configured via QP__DATA_INGEST__POLYGON_API_KEY."
        ),
    )
    polygon_base_url: str = Field(
        default="https://api.polygon.io",
        description="Base URL for Polygon REST API requests.",
    )
    polygon_max_concurrent: int = Field(
        default=4,
        description="Maximum concurrent Polygon intraday historical requests.",
    )
    polygon_min_request_interval_seconds: float = Field(
        default=0.0,
        description="Minimum seconds between Polygon REST requests; useful for free-tier pacing.",
    )
    polygon_timeout_seconds: float = Field(
        default=30.0,
        description="HTTP timeout for Polygon intraday historical requests.",
    )
    sec_user_agent: str = Field(
        default="",
        description=(
            "Required SEC EDGAR User-Agent for governed filing ingestion. "
            "Configure as QP__DATA_INGEST__SEC_USER_AGENT with a contact email."
        ),
    )
    max_bar_age_minutes: int = Field(
        default=60,
        description=(
            "Maximum age in minutes for cached intraday market bars before "
            "PollingMarketDataProvider raises DataStalenessError instead of "
            "returning the stale cached value.  Set 0 to disable the check. "
            "Configured via QP__DATA_INGEST__MAX_BAR_AGE_MINUTES."
        ),
    )
    daily_max_bar_age_minutes: int = Field(
        default=2880,
        description=(
            "Maximum age in minutes for cached daily market bars before "
            "PollingMarketDataProvider raises DataStalenessError. Set 0 to "
            "disable the check. Configured via "
            "QP__DATA_INGEST__DAILY_MAX_BAR_AGE_MINUTES."
        ),
    )

    @model_validator(mode="after")
    def _tiingo_token_required_when_selected(self) -> DataIngestSettings:
        _known_vendors: frozenset[str] = frozenset({"tiingo", "polygon"})
        # Validate chain entries
        for v in self.bar_fetch_fallback_chain:
            if v not in _known_vendors:
                raise ValueError(
                    f"bar_fetch_fallback_chain contains unknown vendor '{v}'. "
                    f"Allowed: {sorted(_known_vendors)}."
                )
        # Resolve effective vendor set (chain supersedes single fallback)
        effective: list[str] = (
            self.bar_fetch_fallback_chain
            if self.bar_fetch_fallback_chain
            else ([self.bar_fetch_fallback] if self.bar_fetch_fallback != "none" else [])
        )
        if "tiingo" in effective and not self.tiingo_api_token:
            raise ValueError(
                "tiingo_api_token must be set when 'tiingo' is in the fallback chain. "
                "Set QP__DATA_INGEST__TIINGO_API_TOKEN."
            )
        if "polygon" in effective and not self.polygon_api_key.strip():
            raise ValueError(
                "polygon_api_key must be set when 'polygon' is in the fallback chain. "
                "Set QP__DATA_INGEST__POLYGON_API_KEY."
            )
        if self.polygon_max_concurrent <= 0:
            raise ValueError("polygon_max_concurrent must be positive")
        if self.polygon_min_request_interval_seconds < 0:
            raise ValueError("polygon_min_request_interval_seconds must be >= 0")
        if self.polygon_timeout_seconds <= 0:
            raise ValueError("polygon_timeout_seconds must be positive")
        if self.max_bar_age_minutes < 0:
            raise ValueError("max_bar_age_minutes must be >= 0")
        if self.daily_max_bar_age_minutes < 0:
            raise ValueError("daily_max_bar_age_minutes must be >= 0")
        return self
