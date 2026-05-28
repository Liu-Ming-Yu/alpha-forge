"""Forecast source selection and promoted-feature policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_platform.config import PlatformSettings


_ALLOWED_FORECAST_SOURCES = {"text", "event", "intraday", "xgboost"}


def _linear_source_policy(
    settings: PlatformSettings,
    source: str,
) -> tuple[str, str, dict[str, float]]:
    if source == "text":
        feature_set_version = str(settings.llm.text_feature_set_version)
        return (
            feature_set_version,
            f"{feature_set_version}:text",
            {
                str(name): float(weight)
                for name, weight in settings.llm.text_feature_weights.items()
            },
        )
    if source == "event":
        feature_set_version = str(settings.alpha.event_feature_set_version)
        return (
            feature_set_version,
            f"{feature_set_version}:event",
            {
                str(name): float(weight)
                for name, weight in settings.alpha.event_feature_weights.items()
            },
        )
    if source == "intraday":
        feature_set_version = str(settings.alpha.intraday_feature_set_version)
        return (
            feature_set_version,
            f"{feature_set_version}:intraday",
            {
                str(name): float(weight)
                for name, weight in settings.alpha.intraday_feature_weights.items()
            },
        )
    raise ValueError(f"unsupported forecast source: {source}")


def _normalise_sources(sources: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for raw in sources:
        source = str(raw).strip()
        if source not in _ALLOWED_FORECAST_SOURCES:
            raise ValueError(f"unsupported forecast source: {source}")
        if source not in seen:
            seen.append(source)
    if not seen:
        raise ValueError("at least one --source is required")
    return tuple(seen)
