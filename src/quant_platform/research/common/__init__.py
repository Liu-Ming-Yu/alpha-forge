"""Composition-bound helpers for research operations.

The pure helpers (result wrapping, instrument-contract loading, calibration-
artifact discovery, durable-input guards, sample-result payload shaping) live
in :mod:`quant_platform.application.research.common` and are re-exported here
for the research composition package. This module itself only carries the
helpers that require composition or infrastructure seams: session/sample
building, Postgres schema verification, and intraday feature loading.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from quant_platform.application.research.common import (
    _BACKTEST_WARMUP_DAYS as _BACKTEST_WARMUP_DAYS,
)
from quant_platform.application.research.common import (
    _instrument_lookup_from_contracts as _instrument_lookup_from_contracts,
)
from quant_platform.application.research.common import (
    _json_default as _json_default,
)
from quant_platform.application.research.common import (
    _latest_calibration_artifact as _latest_calibration_artifact,
)
from quant_platform.application.research.common import (
    _load_calibration_recommendation_bps as _load_calibration_recommendation_bps,
)
from quant_platform.application.research.common import (
    _load_instrument_contracts as _load_instrument_contracts,
)
from quant_platform.application.research.common import (
    _parse_intraday_decision_times as _parse_intraday_decision_times,
)
from quant_platform.application.research.common import (
    _require_durable_research_inputs as _require_durable_research_inputs,
)
from quant_platform.application.research.common import (
    _samples_result_payload as _samples_result_payload,
)
from quant_platform.application.research.common import (
    research_json_result as research_json_result,
)
from quant_platform.bootstrap.data import load_intraday_feature_series
from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema
from quant_platform.bootstrap.session.public_api import create_paper_session
from quant_platform.bootstrap.signal_models import build_default_signal_model

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime
    from pathlib import Path

    from quant_platform.config import PlatformSettings

__all__ = [
    "_BACKTEST_WARMUP_DAYS",
    "_build_samples_to_path",
    "_instrument_lookup_from_contracts",
    "_json_default",
    "_latest_calibration_artifact",
    "_load_calibration_recommendation_bps",
    "_load_instrument_contracts",
    "_load_intraday_feature_series",
    "_parse_intraday_decision_times",
    "_require_durable_research_inputs",
    "_samples_result_payload",
    "_verify_postgres_schema_if_configured",
    "research_json_result",
]


async def _load_intraday_feature_series(
    settings: PlatformSettings,
    contracts: Mapping[uuid.UUID, dict[str, object]],
    feature_set_version: str,
    decision_times: tuple[datetime, ...],
) -> tuple[
    dict[datetime, dict[uuid.UUID, dict[str, float]]],
    dict[datetime, datetime],
]:
    return await load_intraday_feature_series(
        settings,
        contracts,
        feature_set_version,
        decision_times,
    )


async def _verify_postgres_schema_if_configured(settings: PlatformSettings) -> None:
    await verify_postgres_schema(settings)


async def _build_samples_to_path(
    *,
    settings: PlatformSettings,
    contracts_file: str,
    start: datetime,
    end: datetime,
    output: Path,
    feature_set_version: str,
    horizon_days: int,
    bar_seconds: int,
    max_feature_age_days: int,
    date_policy: str = "nyse-sessions",
) -> tuple[Path, Any]:
    """Build supervised samples and write them to ``output``."""
    from quant_platform.services.research_service.sampling.samples import (
        build_supervised_samples,
        research_as_of_dates,
        write_samples_json,
    )

    contracts = _load_instrument_contracts(contracts_file)
    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("0"),
        instrument_contracts=contracts,
        signal_model=build_default_signal_model(settings),
    )
    as_of_dates = research_as_of_dates(start, end, date_policy=date_policy)
    result = await build_supervised_samples(
        feature_repo=session.feature_repo,
        bar_store=session.bar_store,
        instrument_ids=contracts,
        feature_set_version=feature_set_version,
        as_of_dates=as_of_dates,
        horizon_days=horizon_days,
        bar_seconds=bar_seconds,
        max_feature_age_days=max_feature_age_days,
        date_policy=date_policy,
    )
    path = write_samples_json(result.samples, output)
    return path, result
