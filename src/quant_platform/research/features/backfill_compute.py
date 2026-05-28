"""Durable feature-set backfill, dispatched through the typed feature registry."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from quant_platform.bootstrap.data.feature_plugins import build_feature_registry
from quant_platform.core.domain.research import FeatureRequest
from quant_platform.core.domain.signals.feature_inputs import (
    BARS_EOD_INPUT,
    CLOSE_SERIES_INPUT,
    EVENTS_BY_INSTRUMENT_INPUT,
    FeatureInputContext,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.core.domain.market_data import MarketBar


async def backfill_ohlcv_feature_set(
    *,
    bar_history: Mapping[uuid.UUID, Sequence[MarketBar]],
    feature_repo: FeatureRepository,
    strategy_run_id: uuid.UUID,
    as_of: datetime,
    feature_set_version: str,
    artifact_uri: str,
    dry_run: bool,
    events_by_instrument: Mapping[uuid.UUID, Sequence[datetime]] | None = None,
) -> int | None:
    """Backfill one feature-set version for one ``as_of`` through the registry.

    Returns the number of feature vectors produced, or ``None`` when the
    requested ``feature_set_version`` is not a registered feature family.
    """
    registry = build_feature_registry(feature_repo)
    family = registry.family_for_version(feature_set_version)
    if family is None:
        return None

    plugin = registry.get(feature_family=family, feature_set_version=feature_set_version)
    required_input = plugin.required_inputs[0] if plugin.required_inputs else BARS_EOD_INPUT
    payloads: dict[str, object] = {}
    if required_input == CLOSE_SERIES_INPUT:
        payloads[CLOSE_SERIES_INPUT] = {
            instrument_id: [float(bar.close) for bar in bars]
            for instrument_id, bars in bar_history.items()
        }
    else:
        payloads[BARS_EOD_INPUT] = dict(bar_history)
    if events_by_instrument is not None:
        payloads[EVENTS_BY_INSTRUMENT_INPUT] = events_by_instrument

    request = FeatureRequest(
        feature_set_version=feature_set_version,
        instruments=tuple(bar_history),
        start=as_of - timedelta(days=1),
        end=as_of,
        as_of=as_of,
        strategy_run_id=strategy_run_id,
        artifact_uri=artifact_uri,
        context=FeatureInputContext(
            available_inputs=(required_input,),
            payloads=payloads,
        ),
    )
    result = await registry.compute(feature_family=family, request=request)
    if not result.passed:
        return 0
    if not dry_run:
        for vector in result.vectors:
            await feature_repo.store_vector(vector)
    return len(result.vectors)


__all__ = ["backfill_ohlcv_feature_set"]
