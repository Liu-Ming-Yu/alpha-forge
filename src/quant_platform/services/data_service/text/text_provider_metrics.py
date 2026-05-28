"""Best-effort metrics helpers for text source providers."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime


def observe_ingestion_lag(provider_name: str, occurred_at: datetime) -> None:
    """Record text ingestion lag without making metrics a hard dependency."""
    with contextlib.suppress(Exception):
        from quant_platform.telemetry.metrics import set_alpha_text_ingestion_lag

        set_alpha_text_ingestion_lag(
            provider_name,
            (datetime.now(tz=UTC) - occurred_at).total_seconds(),
        )


__all__ = ["observe_ingestion_lag"]
