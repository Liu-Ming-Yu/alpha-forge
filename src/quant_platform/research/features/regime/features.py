"""``regime-v1`` feature factory.

Emits a regime label per date (derived from the bars panel via the
existing :mod:`quant_platform.core.regime` machinery) plus a curated
set of regime × base-feature interactions at the
``(instrument_id, date)`` grain.

The interactions are the columns that actually carry signal: a raw
regime indicator like ``is_risk_on`` is cross-sectionally constant
(every instrument on date ``D`` shares the same value), so its IC is
identically zero. ``momentum × is_risk_on`` varies cross-sectionally
(because momentum does) AND has stable IC *within* the RISK_ON
regime — exactly the property the IC-weighted ranker needs to rotate
weights by regime.

Interactions can be emitted with positive or negated orientation
(see :class:`RegimeInteractionSpec` in ``config.py``); the negated
"anti-" variants exist so the IC-weighted ranker (which runs with
``non_negative=True`` and clips negative-IC features to weight 0) can
express sign-flip priors like "momentum reverses in stress".

See ADR-005 for the full design framing.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pandas as pd

from quant_platform.core.regime import (
    DEFAULT_REGIME_THRESHOLDS,
    MarketRegimeDetector,
    compute_market_stats,
    detector_version,
)
from quant_platform.research.features.contracts import FeatureFrame, FeatureSpec
from quant_platform.research.features.regime.config import (
    BREADTH_SOURCE_ID,
    DEFAULT_CONFIG,
    DEFAULT_INTERACTIONS,
    FEATURE_SET_VERSION,
    INDEX_PROXY_ID,
    REGIME_INDICATOR_LABELS,
    REGIME_STAT_COLUMNS,
    RegimeFeatureConfig,
    RegimeInteractionSpec,
)
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np

    from quant_platform.core.regime import RegimeThresholds

#: Bars and the base-feature panel are merged on ``(instrument_id, date)``.
#: Bars provide the close prices we need to compute the index-proxy
#: market stats; the base-feature panel provides the columns we
#: multiply with regime indicators to produce interactions.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = ("instrument_id", "date", "close")


def _regime_indicator_specs(version: str, labels: Sequence[str]) -> tuple[FeatureSpec, ...]:
    """Specs for the standalone regime-indicator columns.

    These are diagnostic; their IC is identically zero because they
    are cross-sectionally constant on each date. The ranker will
    weight them at 0; they remain in the panel so a reviewer can see
    which regime applied per date without re-running the detector.
    """
    return tuple(
        FeatureSpec(
            name=f"is_regime_{label}",
            family="regime",
            description=(
                f"Indicator (0/1) for the {label.upper()} market regime, "
                "as classified by core.regime.MarketRegimeDetector against "
                "the universe-mean index proxy + per-instrument breadth. "
                "Date-keyed (constant across instruments on the same date) "
                "— surfaced for diagnostics; ranker IC is identically 0 "
                "because the cross-section is uniform. The signal in this "
                "family lives in the regime × base-feature interactions."
            ),
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=200,  # detector's trend_window default
            version=version,
            signal_timestamp="eod_after_close",
            larger_is_better=False,
        )
        for label in labels
    )


def _regime_stat_specs(version: str) -> tuple[FeatureSpec, ...]:
    """Specs for ``regime_trend_z`` / ``_realized_vol`` / ``_breadth``.

    These are the underlying market stats the detector consumes;
    surfaced for two reasons: (a) audit — a reviewer can see *why*
    a given regime label fired on each date; (b) future Shape C work
    may use them as continuous routing keys.

    Same cross-sectional-constant property as the indicators, so IC
    is 0 for these columns standalone. They're not in the curated
    interaction set today.
    """
    descriptions = {
        "regime_trend_z": (
            "Z-score of the universe-mean index close vs its 200d SMA. "
            "Input to the regime classifier; surfaced for audit."
        ),
        "regime_realized_vol": (
            "Annualised 21d realised volatility of the universe-mean index. "
            "Input to the regime classifier."
        ),
        "regime_breadth": (
            "Fraction of instruments above their own 50d SMA on this date. "
            "Input to the regime classifier."
        ),
    }
    return tuple(
        FeatureSpec(
            name=name,
            family="regime",
            description=descriptions[name],
            expected_direction="unknown",
            required_inputs=("close",),
            point_in_time=True,
            lookback_days=200,
            version=version,
            signal_timestamp="eod_after_close",
            larger_is_better=False,
        )
        for name in REGIME_STAT_COLUMNS
    )


def _interaction_specs(
    version: str, interactions: Sequence[RegimeInteractionSpec]
) -> tuple[FeatureSpec, ...]:
    """Specs for the regime × base-feature interaction columns.

    Each interaction's expected direction follows the base feature's
    direction in the regime it survives in — but we cannot know the
    sign a priori. The :class:`RegimeInteractionSpec.negate` flag
    encodes the curation's prior (e.g., momentum reverses in
    risk-off, so we emit the anti-variant); the IC-weighted ranker
    confirms the prior by giving a positive weight (or refutes it by
    clipping to zero).
    """
    specs = []
    for interaction in interactions:
        base_name = interaction.base_feature
        regime_label = interaction.regime_label
        col_name = interaction.column_name
        if interaction.negate:
            description = (
                f"Negated (anti-) regime × base-feature interaction: "
                f"``-{base_name}`` multiplied by the indicator for the "
                f"{regime_label.upper()} regime. Zero on dates outside the "
                f"regime; equals ``-{base_name}`` on dates inside it. "
                "Emitted as a separate column from the positive-orientation "
                f"``{base_name}__x__{regime_label}`` because the IC-weighted "
                "ranker uses ``non_negative=True`` and cannot represent a "
                "sign flip via a single column — curation expects this "
                f"base to *reverse* in {regime_label.upper()}, and only "
                "the anti-variant can carry positive IC in that case."
            )
        else:
            description = (
                f"Regime × base-feature interaction: ``{base_name}`` "
                f"multiplied by the indicator for the {regime_label.upper()} "
                "regime. Zero on dates outside the regime; equals "
                f"``{base_name}`` on dates inside it. The IC-weighted "
                "ranker can weight this differently from the unconditioned "
                f"``{base_name}``, achieving regime-conditional feature "
                "rotation without a per-regime model dispatch."
            )
        specs.append(
            FeatureSpec(
                name=col_name,
                family="regime",
                description=description,
                expected_direction="unknown",
                required_inputs=(base_name, "close"),
                point_in_time=True,
                lookback_days=252,  # at least as long as the longest base
                version=version,
                signal_timestamp="eod_after_close",
                larger_is_better=False,
            )
        )
    return tuple(specs)


def _classify_per_date(
    bars: pd.DataFrame,
    config: RegimeFeatureConfig,
) -> pd.DataFrame:
    """Compute a regime label + supporting stats for each unique date.

    Walks the date axis in order, feeding the canonical
    :func:`compute_market_stats` output into a *single shared*
    :class:`MarketRegimeDetector` via its synchronous
    :meth:`MarketRegimeDetector.step` entrypoint. This is the only
    way to claim research/live parity: the live detector maintains a
    deque-of-candidate-labels and a ``_stable_regime`` that only
    advances after ``stability_window`` matching candidates, plus a
    ``disagree_haircut`` when the stable label and the fresh candidate
    differ. Threading ``current_label`` into ``classify_regime``
    directly (the v6 implementation) skips that state machine and
    would silently produce a *different* label sequence than live
    execution.

    The naive O(N_dates × N_instruments) cost is amortised by
    pre-building per-instrument close arrays once and slicing them
    by date index — universe-300 × 1106 dates runs in well under a
    minute, dwarfed by walk-forward fitting time.

    Subtle correctness reason for *not* vectorising the stats here:
    ``core.algorithms.price_factors.trend_z_score`` is
    ``(close - SMA) / SMA`` — a percentage deviation, NOT a real
    z-score in standard-deviation units. A naive pandas
    ``.rolling().std()`` reimplementation would produce different
    numbers and would silently diverge research evidence from live
    classification. Reuse the canonical function.
    """
    from datetime import UTC  # noqa: PLC0415

    bars_sorted = bars.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    dates = pd.Series(bars_sorted["date"].unique()).sort_values().reset_index(drop=True)

    # Universe-mean index proxy per date — the detector consumes
    # this as a single time series.
    index_per_date = bars_sorted.groupby("date", sort=True)["close"].mean().sort_index()
    date_to_index_position = {date: i for i, date in enumerate(index_per_date.index)}
    index_values_full = index_per_date.to_numpy(dtype=float)

    # Pre-build per-instrument (dates, closes) arrays — built once,
    # sliced per date below. ``dates`` is stored as ``DatetimeIndex``
    # rather than ``ndarray`` so the per-date ``searchsorted`` call
    # below stays in pandas' typed comparison path (numpy arrays of
    # mixed-type Timestamps trip on ordering inside searchsorted).
    per_instrument_dates: dict[uuid.UUID, pd.DatetimeIndex] = {}
    per_instrument_closes: dict[uuid.UUID, np.ndarray] = {}
    for inst, group in bars_sorted.groupby("instrument_id", sort=False):
        try:
            inst_uuid = uuid.UUID(str(inst))
        except ValueError:
            continue
        per_instrument_dates[inst_uuid] = pd.DatetimeIndex(group["date"].to_numpy())
        per_instrument_closes[inst_uuid] = group["close"].to_numpy(dtype=float)

    # Single shared detector instance walks the entire date series.
    # Each ``detector.step(stats, as_of)`` call runs the same
    # state-machine logic the live cycle uses, so the resulting label
    # sequence matches live execution date-for-date.
    #
    # ``log_updates=False``: stepping the detector once per date across
    # the whole universe history emits a ``regime_detector.updated``
    # debug line per call, which would dominate the backtest log
    # (~hundreds of thousands of lines for a universe-300 run). The
    # offline feature path doesn't need the per-step signal; live/paper
    # construction sites keep the default so live detection logging is
    # untouched. State-machine semantics are identical regardless.
    detector = MarketRegimeDetector(thresholds=DEFAULT_REGIME_THRESHOLDS, log_updates=False)
    records: list[dict[str, object]] = []

    for date in dates:
        end_idx = date_to_index_position[date] + 1  # inclusive of this date
        index_closes_up_to = index_values_full[:end_idx].tolist()

        # Trim each instrument's closes to bars on/before this date.
        # ``searchsorted`` gives an O(log N_inst_bars) cut per
        # instrument, fast even at universe-300 × 1106.
        per_inst_view: dict[uuid.UUID, list[float]] = {}
        date_ts = pd.Timestamp(date)
        for inst_uuid, inst_dates in per_instrument_dates.items():
            cutoff = int(inst_dates.searchsorted(date_ts, side="right"))
            if cutoff > 0:
                per_inst_view[inst_uuid] = per_instrument_closes[inst_uuid][:cutoff].tolist()

        as_of_ts = pd.Timestamp(date).tz_localize(UTC).to_pydatetime()
        stats = compute_market_stats(
            index_closes=index_closes_up_to,
            instrument_closes=per_inst_view,
            as_of=as_of_ts,
            trend_window=config.trend_window,
            vol_window=config.vol_window,
            breadth_window=config.breadth_window,
        )
        state = detector.step(stats, as_of=as_of_ts)
        records.append(
            {
                "date": date,
                "regime_label": state.regime_label.value,
                "regime_trend_z": stats.trend_z,
                "regime_realized_vol": stats.realized_vol,
                "regime_breadth": stats.breadth,
            }
        )

    return pd.DataFrame(records)


def compute_regime_features(
    bars: pd.DataFrame,
    base_features_panel: pd.DataFrame,
    *,
    config: RegimeFeatureConfig | None = None,
) -> FeatureFrame:
    """Build the regime feature panel for the latest-stack ablation.

    Inputs
    ------
    ``bars`` — daily-bar panel with ``(instrument_id, date, close)``.
        The detector consumes the universe-mean close + per-instrument
        closes; both are derived from this frame.
    ``base_features_panel`` — the pv+formulaic feature panel at
        ``(instrument_id, date)``. The interaction columns are
        produced by multiplying selected columns from this panel by
        the regime indicators.

    Output
    ------
    A :class:`FeatureFrame` keyed on ``(instrument_id, date)`` with:
    * 4 regime indicators (``is_regime_<label>``),
    * 3 regime stats (``regime_trend_z``, ``regime_realized_vol``,
      ``regime_breadth``),
    * the curated set of regime × base-feature interactions (any
      requested interaction whose base column is missing from
      ``base_features_panel`` is silently skipped — non-strict so the
      family composes with smaller upstream panels). Negated /
      anti-orientation interactions emit ``-base × indicator`` and
      carry the ``anti_`` column-name prefix.
    """
    cfg = config or DEFAULT_CONFIG

    if bars.empty:
        # Edge case used by the smoke / unit tests: return an empty
        # frame with the canonical key columns so the downstream
        # merge doesn't blow up.
        empty = pd.DataFrame(
            columns=["instrument_id", "date"],
        ).astype({"instrument_id": "object"})
        return FeatureFrame(
            frame=empty,
            feature_names=(),
            feature_specs={},
            coverage={},
            key_columns=DEFAULT_KEY_COLUMNS,
        )

    # 1. Per-date regime classification — calls the canonical
    # ``compute_market_stats`` per date and feeds it through a
    # shared ``MarketRegimeDetector.step`` so research/live labels
    # match bit-for-bit.
    regime_panel = _classify_per_date(bars, cfg)

    # 2. Indicator columns (0/1 broadcast per date).
    for label in REGIME_INDICATOR_LABELS:
        regime_panel[f"is_regime_{label}"] = (regime_panel["regime_label"] == label).astype(float)
    # The regime_label string column itself is dropped from the
    # output — useful internally but not a feature consumed by the
    # ranker; the indicators carry the information.
    regime_panel = regime_panel.drop(columns=["regime_label"])

    # 3. Broadcast date-keyed regime columns to the (instrument, date)
    # grain by left-joining onto the base panel's date dimension.
    base_keys = base_features_panel[["instrument_id", "date"]].drop_duplicates()
    panel = base_keys.merge(regime_panel, on="date", how="left")

    # 4. Interaction columns: base_col (× ±1 depending on orientation) × is_regime_<label>.
    interaction_names: list[str] = []
    emitted_specs: list[RegimeInteractionSpec] = []
    base_for_join = base_features_panel.set_index(["instrument_id", "date"], drop=False)
    panel_indexed = panel.set_index(["instrument_id", "date"], drop=False)
    for interaction in DEFAULT_INTERACTIONS:
        if interaction.base_feature not in base_features_panel.columns:
            # Non-strict — caller may run with a feature subset; skip
            # the interaction silently rather than fail the whole
            # family. Test coverage pins this behaviour.
            continue
        indicator_col = f"is_regime_{interaction.regime_label}"
        sign = -1.0 if interaction.negate else 1.0
        base_values = base_for_join[interaction.base_feature].reindex(panel_indexed.index)
        panel_indexed[interaction.column_name] = (
            sign * base_values.to_numpy() * panel_indexed[indicator_col].to_numpy()
        )
        interaction_names.append(interaction.column_name)
        emitted_specs.append(interaction)

    panel = panel_indexed.reset_index(drop=True)

    # 5. Assemble the FeatureFrame.
    indicator_names = tuple(f"is_regime_{label}" for label in REGIME_INDICATOR_LABELS)
    stat_names = tuple(REGIME_STAT_COLUMNS)
    interaction_names_tuple = tuple(interaction_names)
    all_feature_names = indicator_names + stat_names + interaction_names_tuple

    specs = (
        *_regime_indicator_specs(cfg.version, REGIME_INDICATOR_LABELS),
        *_regime_stat_specs(cfg.version),
        *_interaction_specs(cfg.version, emitted_specs),
    )
    feature_specs_map = {spec.name: spec for spec in specs}

    output_cols = ["instrument_id", "date", *all_feature_names]
    out_panel = panel[output_cols].copy()
    # Coverage = count of non-null rows per feature, matching the
    # convention used by other families. Conservative — counts NaN
    # interaction rows (where the base feature itself was NaN) as
    # missing coverage, so a base feature with poor coverage doesn't
    # silently inflate the interaction's apparent coverage.
    coverage_map: dict[str, int] = {
        name: int(out_panel[name].notna().sum()) for name in all_feature_names
    }
    return FeatureFrame(
        frame=out_panel,
        feature_names=all_feature_names,
        feature_specs=feature_specs_map,
        coverage=coverage_map,
        key_columns=DEFAULT_KEY_COLUMNS,
    )


def regime_detector_metadata(
    *,
    thresholds: RegimeThresholds | None = None,
) -> Mapping[str, object]:
    """Return a JSON-safe metadata block pinning the regime detector.

    Emitted into per-arm evidence (when the arm uses regime features)
    and into the run manifest (when any arm in the run uses regime
    features). Lets a future audit confirm that the detector
    thresholds / proxy strategy were not silently retuned between the
    rerun that produced the evidence and the rerun the auditor is
    comparing against. Closes ADR-005 action item #10.
    """
    thresh = thresholds or DEFAULT_REGIME_THRESHOLDS
    return {
        "feature_set_version": FEATURE_SET_VERSION,
        "detector_version": detector_version("v1.0-rule-based", thresh),
        "thresholds": {
            "crisis_vol": thresh.crisis_vol,
            "risk_off_vol": thresh.risk_off_vol,
            "low_vol": thresh.low_vol,
            "downtrend_z": thresh.downtrend_z,
            "uptrend_z": thresh.uptrend_z,
            "weak_breadth": thresh.weak_breadth,
            "strong_breadth": thresh.strong_breadth,
            "hysteresis_vol": thresh.hysteresis_vol,
            "stability_window": thresh.stability_window,
        },
        "index_proxy": INDEX_PROXY_ID,
        "breadth_source": BREADTH_SOURCE_ID,
    }


#: Default training feature names: ONLY the interactions, not the raw
#: indicators or stats (those have IC=0 cross-sectionally and would
#: just waste fit-time degrees of freedom). The latest-stack arm H
#: filters its training features to this set; auditors can still see
#: the indicators in the panel for debugging.
DEFAULT_TRAINING_FEATURE_NAMES: tuple[str, ...] = tuple(
    interaction.column_name for interaction in DEFAULT_INTERACTIONS
)


# Specs computed up-front so the manifest at __init__ time is stable.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    *_regime_indicator_specs(FEATURE_SET_VERSION, REGIME_INDICATOR_LABELS),
    *_regime_stat_specs(FEATURE_SET_VERSION),
    *_interaction_specs(FEATURE_SET_VERSION, DEFAULT_INTERACTIONS),
)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)


__all__ = [
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "REQUIRED_INPUT_COLUMNS",
    "compute_regime_features",
    "regime_detector_metadata",
]
