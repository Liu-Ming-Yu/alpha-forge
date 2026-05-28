"""Metadata helpers for feature failure attribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from quant_platform.services.research_service.feature_quality.failures.metrics import ic_summary

if TYPE_CHECKING:
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def direction_by_feature(diagnostics: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Index direction diagnostics by feature name."""
    raw = diagnostics.get("features", ())
    if not isinstance(raw, Sequence):
        return {}
    out: dict[str, Mapping[str, object]] = {}
    for row in raw:
        if isinstance(row, Mapping) and row.get("feature_name") is not None:
            out[str(row["feature_name"])] = row
    return out


def feature_families(
    feature_names: Sequence[str],
    metadata: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    """Load diagnostic feature-family groups and mark unmapped features."""
    raw = metadata.get("families", {})
    families: dict[str, tuple[str, ...]] = {}
    if isinstance(raw, Mapping):
        for family, values in raw.items():
            if isinstance(values, Sequence) and not isinstance(values, str):
                families[str(family)] = tuple(str(value) for value in values)
    mapped = {feature for values in families.values() for feature in values}
    missing = tuple(name for name in feature_names if name not in mapped)
    if missing:
        families["unmapped"] = missing
    return families


def family_rows(
    families: Mapping[str, Sequence[str]],
    feature_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Summarise family-level ranking by official-horizon IC."""
    by_name = {str(row["feature_name"]): row for row in feature_rows}
    rows: list[dict[str, object]] = []
    for family, names in sorted(families.items()):
        ranked = sorted(
            (name for name in names if name in by_name),
            key=lambda name: official_ic(by_name[name]),
            reverse=True,
        )
        rows.append(
            {
                "family": family,
                "features": list(ranked),
                "representative_candidate": ranked[0] if ranked else None,
                "feature_count": len(ranked),
            }
        )
    return rows


def family_best_features(
    families: Mapping[str, Sequence[str]],
    samples_by_horizon: Mapping[int, Sequence[SupervisedAlphaSample]],
    directions: Mapping[str, Mapping[str, object]],
    official_horizon_days: int,
) -> dict[str, str]:
    """Return the strongest official-horizon feature per family."""
    best: dict[str, str] = {}
    samples = samples_by_horizon.get(official_horizon_days, ())
    for family, names in families.items():
        ranked = sorted(
            names,
            key=lambda name: ic_summary(samples, name, recommended_sign(directions.get(name, {})))[
                "ic_mean"
            ],
            reverse=True,
        )
        if ranked:
            best[family] = ranked[0]
    return best


def recommended_sign(row: Mapping[str, object]) -> float:
    """Return the numeric sign implied by direction diagnostics."""
    return -1.0 if str(row.get("recommended_orientation", "positive")) == "negative" else 1.0


def recommended_orientation_payload(row: Mapping[str, object]) -> Mapping[str, object]:
    """Return the selected orientation payload from direction diagnostics."""
    orientations = row.get("orientations", {})
    if not isinstance(orientations, Mapping):
        return {}
    selected = orientations.get(str(row.get("recommended_orientation", "positive")), {})
    return selected if isinstance(selected, Mapping) else {}


def recommended_list(row: Mapping[str, object], key: str) -> list[str]:
    """Return a string list from the selected orientation payload."""
    raw = recommended_orientation_payload(row).get(key, ())
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [str(value) for value in raw]
    return []


def feature_family(feature_name: str, families: Mapping[str, Sequence[str]]) -> str:
    """Return the configured family for one feature."""
    for family, names in families.items():
        if feature_name in names:
            return family
    return "unmapped"


def diagnostic_recommendation(
    *,
    official: Mapping[str, object],
    null: Mapping[str, float],
    feature_name: str,
    family: str,
    family_best: Mapping[str, str],
) -> str:
    """Assign a diagnostics-only recommendation label."""
    observations = _float_value(official.get("ic_observations", 0.0))
    ic_mean = _float_value(official.get("ic_mean", 0.0))
    null_p95 = float(null.get("null_p95", 0.0))
    if observations < 252:
        return "needs_more_data"
    if ic_mean <= 0.0 or ic_mean <= null_p95:
        return "discard"
    if family_best.get(family) == feature_name:
        return "family_representative_candidate"
    return "repair_candidate"


def official_ic(row: Mapping[str, object]) -> float:
    """Read official 21d IC from a feature attribution row."""
    horizons = row.get("horizon_comparison", {})
    if not isinstance(horizons, Mapping):
        return 0.0
    official = horizons.get("21", {})
    if not isinstance(official, Mapping):
        return 0.0
    return float(official.get("ic_mean", 0.0))


def feature_seed(seed: int, feature_name: str) -> int:
    """Build a deterministic seed per feature name."""
    return seed + sum((index + 1) * ord(char) for index, char in enumerate(feature_name))


def _float_value(raw: object, default: float = 0.0) -> float:
    if not isinstance(raw, int | float | str):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError, OverflowError):
        return default


__all__ = [
    "diagnostic_recommendation",
    "direction_by_feature",
    "family_best_features",
    "family_rows",
    "feature_families",
    "feature_family",
    "feature_seed",
    "recommended_list",
    "recommended_sign",
]
