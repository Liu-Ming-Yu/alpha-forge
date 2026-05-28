"""Source-density guards for governed catalyst paper campaigns."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from quant_platform.research.common import _json_default
from quant_platform.services.research_service.features.paper_alpha.text_features import (
    PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    TEXT_CATALYST_V10_ALPHA_FEATURES,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


MIN_SOURCE_EVENTS_PER_INSTRUMENT = 3
MIN_EXHIBIT_EVENTS_PER_INSTRUMENT = MIN_SOURCE_EVENTS_PER_INSTRUMENT
MIN_DAILY_NONZERO_FRACTION = 0.05
REQUIRED_INSTRUMENTS = 15

CATALYST_FEATURES_BY_SET = {
    PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION: TEXT_CATALYST_V10_ALPHA_FEATURES,
}

SOURCE_COUNT_FIELD_BY_SET = {
    PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION: "primary_events_by_symbol",
}


def maybe_block_for_catalyst_source_density(
    *,
    feature_set_version: str,
    source_data_manifest: Path | None,
    samples: Sequence[SupervisedAlphaSample],
    output_root: Path,
    sample_slug: str,
) -> dict[str, object] | None:
    """Return a blocked payload when catalyst source data is too sparse."""
    feature_names = CATALYST_FEATURES_BY_SET.get(feature_set_version)
    if feature_names is None:
        return None
    payload = _catalyst_source_density_payload(
        feature_set_version=feature_set_version,
        feature_names=feature_names,
        source_data_manifest=source_data_manifest,
        samples=samples,
    )
    if payload["passed"]:
        return None
    blocked_root = output_root / "_blocked" / sample_slug
    blocked_root.mkdir(parents=True, exist_ok=True)
    blocked_path = blocked_root / "blocked_source_density_summary.json"
    blocked_path.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload["blocked_source_density_summary"] = str(blocked_path)
    return payload


def _catalyst_source_density_payload(
    *,
    feature_set_version: str,
    feature_names: Sequence[str],
    source_data_manifest: Path | None,
    samples: Sequence[SupervisedAlphaSample],
) -> dict[str, object]:
    blockers: list[str] = []
    manifest_payload: Mapping[str, Any] = {}
    if source_data_manifest is None:
        blockers.append(f"{feature_set_version} requires --source-data-manifest")
    else:
        try:
            loaded = json.loads(source_data_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"source-data manifest unavailable: {exc}")
        else:
            if isinstance(loaded, Mapping):
                manifest_payload = loaded
            else:
                blockers.append("source-data manifest must be a JSON object")

    source_count_field = SOURCE_COUNT_FIELD_BY_SET[feature_set_version]
    source_label = (
        "SEC primary" if source_count_field == "primary_events_by_symbol" else "SEC exhibit"
    )
    source_counts = _source_counts_by_symbol(manifest_payload, count_field=source_count_field)
    covered_symbols = tuple(symbol for symbol, count in source_counts.items() if count > 0)
    if len(covered_symbols) < REQUIRED_INSTRUMENTS:
        blockers.append(
            f"{source_label} coverage {len(covered_symbols)} < {REQUIRED_INSTRUMENTS} instruments"
        )
    thin_symbols = tuple(
        symbol
        for symbol, count in source_counts.items()
        if count < MIN_SOURCE_EVENTS_PER_INSTRUMENT
    )
    if thin_symbols:
        blockers.append(
            f"{source_label} event count below "
            f"{MIN_SOURCE_EVENTS_PER_INSTRUMENT} for: {', '.join(thin_symbols)}"
        )

    nonzero_fractions = _candidate_nonzero_fractions(samples, feature_names=feature_names)
    thin_features = tuple(
        feature
        for feature, fraction in nonzero_fractions.items()
        if fraction < MIN_DAILY_NONZERO_FRACTION
    )
    if thin_features:
        blockers.append(
            "daily catalyst nonzero_fraction below "
            f"{MIN_DAILY_NONZERO_FRACTION:.2f} for: {', '.join(thin_features)}"
        )

    return {
        "passed": not blockers,
        "reason": "catalyst source density blocked paper research campaign"
        if blockers
        else "catalyst source density passed",
        "blockers": blockers,
        "source_data_manifest": str(source_data_manifest) if source_data_manifest else None,
        source_count_field: dict(source_counts),
        "covered_instruments": len(covered_symbols),
        "source_count_field": source_count_field,
        "min_source_events_per_instrument": MIN_SOURCE_EVENTS_PER_INSTRUMENT,
        "min_exhibit_events_per_instrument": MIN_EXHIBIT_EVENTS_PER_INSTRUMENT,
        "candidate_nonzero_fraction": nonzero_fractions,
        "min_daily_nonzero_fraction": MIN_DAILY_NONZERO_FRACTION,
    }


def _source_counts_by_symbol(manifest: Mapping[str, Any], *, count_field: str) -> dict[str, int]:
    requested = []
    download = manifest.get("download")
    if isinstance(download, Mapping):
        raw_requested = download.get("requested_symbols")
        if isinstance(raw_requested, list):
            requested = [str(symbol).upper() for symbol in raw_requested]
    raw_counts = manifest.get(count_field)
    counts = {symbol: 0 for symbol in requested}
    if isinstance(raw_counts, Mapping):
        for symbol, value in raw_counts.items():
            try:
                counts[str(symbol).upper()] = int(value)
            except (TypeError, ValueError):
                counts[str(symbol).upper()] = 0
    return dict(sorted(counts.items()))


def _candidate_nonzero_fractions(
    samples: Sequence[SupervisedAlphaSample],
    *,
    feature_names: Sequence[str],
) -> dict[str, float]:
    total = len(samples)
    if total <= 0:
        return {feature: 0.0 for feature in feature_names}
    fractions: dict[str, float] = {}
    for feature in feature_names:
        nonzero = 0
        for sample in samples:
            try:
                value = float(sample.features.get(feature, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if value != 0.0:
                nonzero += 1
        fractions[feature] = nonzero / float(total)
    return fractions


__all__ = ["maybe_block_for_catalyst_source_density"]
