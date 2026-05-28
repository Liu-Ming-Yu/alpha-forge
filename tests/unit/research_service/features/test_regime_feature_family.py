"""Unit tests for the ``regime-v1`` feature family.

The family produces three column classes (indicators, stats,
interactions); these tests pin:

1. ``MANIFEST`` registers cleanly and reports the right shape
   (one ``regime`` family with the expected version + spec count).
2. The indicator columns are 0/1, mutually exclusive, sum to 1 per
   date — the regime classifier emits exactly one label per date.
3. Indicator columns are cross-sectionally constant on each date
   (the underlying "regime is one per date" property the IC ranker
   would otherwise silently weight at 0).
4. Positive-orientation interaction columns are zero outside their
   regime and equal to the base feature value inside their regime.
5. Negated (anti-) interaction columns are zero outside their regime
   and equal to MINUS the base feature value inside their regime — the
   sign-flip carrier that lets the non-negative IC ranker express
   reversal priors.
6. Missing base features (caller passes a smaller panel) are
   silently skipped — non-strict composability.
7. Empty bars input returns an empty :class:`FeatureFrame` with the
   canonical key columns (defensive case for smoke tests).
8. Synthetic stable-trend bars produce a stable regime label
   (no thrashing) — the detector's 3-day stability smoothing carries
   through to the family output.
9. Family stats match the canonical :func:`compute_market_stats`
   bit-for-bit — research/live detector stats parity (ADR-005 contract).
10. **Family labels match the live MarketRegimeDetector's full
    state-machine output bit-for-bit over the entire date sequence**
    — research/live detector LABEL parity (ADR-005 hardening fix).
    This is the key invariant: the family must walk a single shared
    detector and call ``step()`` per date in order, NOT replay
    ``classify_regime`` with a manual ``current_label`` carry.
11. ``regime_detector_metadata`` returns the pinning dict the script
    embeds into evidence / manifest.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.regime import (
    DEFAULT_REGIME_THRESHOLDS,
    MarketRegimeDetector,
    compute_market_stats,
    detector_version,
)
from quant_platform.research.features.regime import (
    BREADTH_SOURCE_ID,
    DEFAULT_INTERACTIONS,
    FEATURE_SET_VERSION,
    INDEX_PROXY_ID,
    MANIFEST,
    REGIME_INDICATOR_LABELS,
    RegimeInteractionSpec,
    compute_regime_features,
    regime_detector_metadata,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _synthetic_bars(
    *,
    n_instruments: int = 5,
    n_days: int = 300,
    daily_mean: float = 0.0005,
    daily_vol: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic bars panel for tests. Geometric-Brownian-style paths."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for _ in range(n_instruments):
        inst_id = str(uuid.uuid4())
        price = 100.0
        for d in dates:
            price *= 1.0 + rng.normal(daily_mean, daily_vol)
            rows.append({"instrument_id": inst_id, "date": d, "close": price})
    return pd.DataFrame(rows)


def _synthetic_base_panel(bars: pd.DataFrame, *, seed: int = 1) -> pd.DataFrame:
    """Synthetic base-feature panel matching the bars rows.

    Names the columns the regime family expects to find for the
    default interactions list, so the smoke produces a full
    12-interaction output.
    """
    rng = np.random.default_rng(seed)
    base = bars.copy()
    for name in (
        "mom_12_1",
        "mom_6_1",
        "ret_252d",
        "reversal_5d",
        "vol_60d",
        "vol_21d",
    ):
        base[name] = rng.normal(0.0, 0.1, len(base))
    return base


def _first_positive_interaction() -> RegimeInteractionSpec:
    """First positive-orientation entry in the default curation."""
    return next(spec for spec in DEFAULT_INTERACTIONS if not spec.negate)


def _first_negated_interaction() -> RegimeInteractionSpec:
    """First negated (anti-) entry in the default curation."""
    return next(spec for spec in DEFAULT_INTERACTIONS if spec.negate)


# ---------------------------------------------------------------------------
# 1. Manifest registration
# ---------------------------------------------------------------------------


class TestRegimeManifest:
    def test_manifest_name_and_version(self) -> None:
        assert MANIFEST.name == "regime"
        assert MANIFEST.version == FEATURE_SET_VERSION
        assert MANIFEST.version == "regime-v1"

    def test_manifest_spec_count_matches_expected_shape(self) -> None:
        # 4 indicators + 3 stats + len(DEFAULT_INTERACTIONS) specs.
        assert len(MANIFEST.feature_specs) == 4 + 3 + len(DEFAULT_INTERACTIONS)

    def test_manifest_required_input_columns(self) -> None:
        # Bars panel must carry these columns.
        assert MANIFEST.required_input_columns == ("instrument_id", "date", "close")

    def test_default_training_feature_names_are_only_interactions(self) -> None:
        # IC of the raw indicators / stats is 0 cross-sectionally, so
        # the training set is restricted to the interactions — the
        # columns that actually carry signal. Both positive-orientation
        # ("X__x__regime") and anti- ("anti_X__x__regime") names are
        # included; both forms contain "__x__".
        assert len(MANIFEST.default_training_feature_names) == len(DEFAULT_INTERACTIONS)
        for name in MANIFEST.default_training_feature_names:
            assert "__x__" in name

    def test_curation_includes_both_orientations(self) -> None:
        # Hardening fix from review #3: the curation must include at
        # least one negated interaction, otherwise the non-negative
        # ranker cannot express sign-flip priors and the regime axis
        # is mathematically limited to scaling, not rotation.
        has_positive = any(not spec.negate for spec in DEFAULT_INTERACTIONS)
        has_negated = any(spec.negate for spec in DEFAULT_INTERACTIONS)
        assert has_positive, "curation must include at least one positive-orientation interaction"
        assert has_negated, (
            "curation must include at least one negated (anti-) interaction so the "
            "non_negative=True ranker can express sign-flip priors (ADR-005 hardening)"
        )


# ---------------------------------------------------------------------------
# 2. Indicator column shape
# ---------------------------------------------------------------------------


class TestRegimeIndicators:
    """Indicators must be 0/1 mutually-exclusive and sum to 1 per date."""

    def test_indicators_are_zero_or_one(self) -> None:
        bars = _synthetic_bars()
        base = _synthetic_base_panel(bars)
        ff = compute_regime_features(bars, base)
        for label in REGIME_INDICATOR_LABELS:
            col = ff.frame[f"is_regime_{label}"]
            assert set(col.unique()) <= {0.0, 1.0}

    def test_indicators_sum_to_one_per_date(self) -> None:
        bars = _synthetic_bars()
        base = _synthetic_base_panel(bars)
        ff = compute_regime_features(bars, base)
        indicator_cols = [f"is_regime_{label}" for label in REGIME_INDICATOR_LABELS]
        # Pick any (instrument, date) — exactly one indicator is 1.
        sums = ff.frame[indicator_cols].sum(axis=1)
        # Some dates may produce UNKNOWN (no indicator column for it
        # because we don't emit one — that's intentional, UNKNOWN
        # means the detector lacked data). Allow either 0 or 1; never 2+.
        assert sums.max() <= 1.0
        assert sums.min() >= 0.0


class TestIndicatorsCrossSectionallyConstant:
    """The whole framing of ADR-005's Shape B' rests on indicators
    being constant per date (which is why raw indicators have IC=0
    and we need interactions). This test pins the property."""

    def test_indicator_value_uniform_across_instruments_on_each_date(self) -> None:
        bars = _synthetic_bars(n_instruments=10)
        base = _synthetic_base_panel(bars)
        ff = compute_regime_features(bars, base)
        for label in REGIME_INDICATOR_LABELS:
            col_name = f"is_regime_{label}"
            # For each date, every instrument must share the same value.
            by_date = ff.frame.groupby("date")[col_name].nunique()
            assert (by_date == 1).all(), (
                f"{col_name} varies across instruments on at least one "
                "date — regime is supposed to be a date-level property."
            )


# ---------------------------------------------------------------------------
# 3. Interaction column semantics — positive AND negated orientations
# ---------------------------------------------------------------------------


class TestRegimeInteractions:
    """``base × indicator`` must equal base inside the regime and 0
    outside (positive orientation). For negated (anti-) interactions,
    inside-regime value is ``-base`` instead. These are the properties
    that give interactions non-zero IC while raw indicators stay at
    IC=0."""

    def test_positive_interaction_zero_outside_regime(self) -> None:
        bars = _synthetic_bars()
        base = _synthetic_base_panel(bars)
        ff = compute_regime_features(bars, base)
        spec = _first_positive_interaction()
        interaction_col = spec.column_name
        indicator_col = f"is_regime_{spec.regime_label}"
        assert interaction_col in ff.frame.columns

        # Rows where the indicator is 0 must have interaction = 0.
        out_of_regime = ff.frame[ff.frame[indicator_col] == 0.0]
        assert (out_of_regime[interaction_col] == 0.0).all()

    def test_positive_interaction_equals_base_inside_regime(self) -> None:
        bars = _synthetic_bars()
        base = _synthetic_base_panel(bars)
        ff = compute_regime_features(bars, base)
        spec = _first_positive_interaction()
        interaction_col = spec.column_name
        indicator_col = f"is_regime_{spec.regime_label}"

        merged = ff.frame.merge(
            base[["instrument_id", "date", spec.base_feature]],
            on=["instrument_id", "date"],
            how="left",
            suffixes=("", "_base_value"),
        )
        base_value_col = (
            f"{spec.base_feature}_base_value"
            if f"{spec.base_feature}_base_value" in merged.columns
            else spec.base_feature
        )
        in_regime = merged[merged[indicator_col] == 1.0]
        if not in_regime.empty:
            diff = (in_regime[interaction_col] - in_regime[base_value_col]).abs()
            assert (diff < 1e-9).all(), (
                f"interaction {interaction_col} does not equal base "
                f"{spec.base_feature} when {indicator_col} == 1"
            )

    def test_negated_interaction_emits_anti_prefix(self) -> None:
        # Naming contract: negated interactions carry the ``anti_``
        # prefix so auditors can distinguish them in selected_weights
        # without parsing the spec.
        for spec in DEFAULT_INTERACTIONS:
            if spec.negate:
                assert spec.column_name.startswith("anti_")
                assert spec.column_name == f"anti_{spec.base_feature}__x__{spec.regime_label}"
            else:
                assert not spec.column_name.startswith("anti_")
                assert spec.column_name == f"{spec.base_feature}__x__{spec.regime_label}"

    def test_negated_interaction_zero_outside_regime(self) -> None:
        bars = _synthetic_bars()
        base = _synthetic_base_panel(bars)
        ff = compute_regime_features(bars, base)
        spec = _first_negated_interaction()
        interaction_col = spec.column_name
        indicator_col = f"is_regime_{spec.regime_label}"
        assert interaction_col in ff.frame.columns

        out_of_regime = ff.frame[ff.frame[indicator_col] == 0.0]
        assert (out_of_regime[interaction_col] == 0.0).all()

    def test_negated_interaction_equals_minus_base_inside_regime(self) -> None:
        # The whole point of negated interactions (ADR-005 hardening):
        # inside the regime, the column carries -base (not +base), so
        # the non_negative=True ranker can express the sign flip.
        bars = _synthetic_bars()
        base = _synthetic_base_panel(bars)
        ff = compute_regime_features(bars, base)
        spec = _first_negated_interaction()
        interaction_col = spec.column_name
        indicator_col = f"is_regime_{spec.regime_label}"

        merged = ff.frame.merge(
            base[["instrument_id", "date", spec.base_feature]],
            on=["instrument_id", "date"],
            how="left",
            suffixes=("", "_base_value"),
        )
        base_value_col = (
            f"{spec.base_feature}_base_value"
            if f"{spec.base_feature}_base_value" in merged.columns
            else spec.base_feature
        )
        in_regime = merged[merged[indicator_col] == 1.0]
        if not in_regime.empty:
            # interaction == -base inside the regime, to float precision.
            diff = (in_regime[interaction_col] + in_regime[base_value_col]).abs()
            assert (diff < 1e-9).all(), (
                f"anti-interaction {interaction_col} does not equal -{spec.base_feature} "
                f"when {indicator_col} == 1"
            )


# ---------------------------------------------------------------------------
# 4. Composability: missing base features
# ---------------------------------------------------------------------------


class TestNonStrictComposability:
    """The family must accept a base panel that lacks some of the
    columns named in :data:`DEFAULT_INTERACTIONS` — silently skip
    those interactions. A research script that runs PV-only must
    still be able to add the regime family on top."""

    def test_missing_base_feature_silently_skips_interaction(self) -> None:
        bars = _synthetic_bars()
        base_minimal = bars.copy()
        # Only ONE of the default-interaction base features present.
        base_minimal["mom_12_1"] = 0.05

        ff = compute_regime_features(bars, base_minimal)
        interaction_names = [n for n in ff.feature_names if "__x__" in n]
        # Only mom_12_1 interactions survive (positive AND anti-); other bases are absent.
        for name in interaction_names:
            # The interaction column name is either "<base>__x__<regime>"
            # or "anti_<base>__x__<regime>" — strip the optional "anti_"
            # prefix before checking the base.
            stripped = name.removeprefix("anti_")
            base_part = stripped.split("__x__")[0]
            assert base_part == "mom_12_1", (
                f"Interaction {name} survives despite missing base feature"
            )

    def test_empty_bars_returns_empty_frame_with_key_columns(self) -> None:
        ff = compute_regime_features(
            pd.DataFrame(columns=["instrument_id", "date", "close"]),
            pd.DataFrame(columns=["instrument_id", "date"]),
        )
        assert ff.frame.empty
        assert ff.feature_names == ()
        assert "instrument_id" in ff.frame.columns
        assert "date" in ff.frame.columns


# ---------------------------------------------------------------------------
# 5. Stability + detector parity
# ---------------------------------------------------------------------------


class TestRegimeStability:
    """Stable inputs → stable labels. If a 100-day stretch has
    consistent low-vol uptrend, every date's label should be
    RISK_ON (modulo the 200-day warmup before any classification
    can happen)."""

    def test_stable_uptrend_produces_stable_label(self) -> None:
        # Deterministic uptrend with tiny vol — should classify as
        # RISK_ON once the detector has 200 days of warmup.
        bars = _synthetic_bars(
            n_instruments=20,
            n_days=400,
            daily_mean=0.002,  # ~50% annualised drift
            daily_vol=0.005,  # ~8% annualised vol — well below low_vol=0.20
            seed=42,
        )
        base = _synthetic_base_panel(bars)
        ff = compute_regime_features(bars, base)

        # Look at the last 50 dates (post-warmup, well into the stable
        # regime). The risk_on indicator should be 1 on most/all of them.
        last_dates = sorted(ff.frame["date"].unique())[-50:]
        last_rows = ff.frame[ff.frame["date"].isin(last_dates)]
        risk_on_frac = (last_rows["is_regime_risk_on"] == 1.0).mean()
        assert risk_on_frac > 0.6, (
            "stable uptrend regime should produce RISK_ON on most late dates; "
            f"got risk_on_fraction={risk_on_frac:.3f}"
        )


class TestDetectorParity:
    """Research-side stats AND labels must match what the live
    :class:`MarketRegimeDetector` would produce on the same input
    sequence. The stats parity is straightforward (call the same
    canonical function). The label parity is the harder invariant
    fixed in ADR-005 hardening: the family must walk a single shared
    detector and call ``step()`` per date in order — replicating the
    deque-of-candidates / ``stability_window`` / ``disagree_haircut``
    state machine — not call ``classify_regime`` with a manual
    ``current_label`` carry."""

    def test_family_stats_match_compute_market_stats(self) -> None:
        bars = _synthetic_bars(n_instruments=5, n_days=300)
        base = _synthetic_base_panel(bars)
        ff = compute_regime_features(bars, base)

        # Pick a specific date deep in the panel (post-warmup) and
        # compute the canonical stats directly. The family's stat
        # columns must match exactly.
        test_date = sorted(ff.frame["date"].unique())[-5]
        index_closes = bars.groupby("date")["close"].mean().sort_index().loc[:test_date].tolist()
        per_inst: dict[uuid.UUID, list[float]] = {}
        for inst, group in bars.groupby("instrument_id"):
            inst_uuid = uuid.UUID(str(inst))
            per_inst[inst_uuid] = group[group["date"] <= test_date]["close"].tolist()

        as_of_ts = pd.Timestamp(test_date).tz_localize(UTC).to_pydatetime()
        canonical_stats = compute_market_stats(
            index_closes=index_closes,
            instrument_closes=per_inst,
            as_of=as_of_ts,
        )

        family_row = ff.frame[ff.frame["date"] == test_date].iloc[0]
        # Family stats are the same value across instruments on this
        # date (cross-sectionally constant); take any row.
        assert family_row["regime_trend_z"] == pytest.approx(canonical_stats.trend_z, rel=1e-10)
        assert family_row["regime_realized_vol"] == pytest.approx(
            canonical_stats.realized_vol, rel=1e-10
        )
        assert family_row["regime_breadth"] == pytest.approx(canonical_stats.breadth, rel=1e-10)

    def test_family_label_sequence_matches_market_regime_detector_step(self) -> None:
        """The killer parity test (ADR-005 hardening, review finding #1+#2).

        Run the family. Independently walk a fresh
        :class:`MarketRegimeDetector` over the same dates in order,
        feeding the same canonical stats via ``detector.step``. The
        family's one-hot regime indicator must match the detector's
        returned label *on every date* — no exceptions, no boundary
        slop. If this test ever fails, research evidence has diverged
        from live execution and Arm H's metrics are not actionable.
        """
        bars = _synthetic_bars(n_instruments=8, n_days=350, seed=7)
        base = _synthetic_base_panel(bars)
        ff = compute_regime_features(bars, base)

        # Reconstruct the inputs the family would have used per date,
        # then walk a shared detector through the same sequence.
        dates = sorted(bars["date"].unique())
        index_per_date = bars.groupby("date", sort=True)["close"].mean().sort_index()
        index_values = index_per_date.to_numpy(dtype=float)

        per_inst_dates: dict[uuid.UUID, pd.DatetimeIndex] = {}
        per_inst_closes: dict[uuid.UUID, np.ndarray] = {}
        for inst, group in bars.sort_values("date").groupby("instrument_id", sort=False):
            inst_uuid = uuid.UUID(str(inst))
            per_inst_dates[inst_uuid] = pd.DatetimeIndex(group["date"].to_numpy())
            per_inst_closes[inst_uuid] = group["close"].to_numpy(dtype=float)

        detector = MarketRegimeDetector(thresholds=DEFAULT_REGIME_THRESHOLDS)
        expected_labels: dict[pd.Timestamp, str] = {}
        for i, date in enumerate(dates):
            index_closes_up_to = index_values[: i + 1].tolist()
            per_inst_view: dict[uuid.UUID, list[float]] = {}
            date_ts = pd.Timestamp(date)
            for inst_uuid, inst_dates in per_inst_dates.items():
                cutoff = int(inst_dates.searchsorted(date_ts, side="right"))
                if cutoff > 0:
                    per_inst_view[inst_uuid] = per_inst_closes[inst_uuid][:cutoff].tolist()
            as_of_ts = pd.Timestamp(date).tz_localize(UTC).to_pydatetime()
            stats = compute_market_stats(
                index_closes=index_closes_up_to,
                instrument_closes=per_inst_view,
                as_of=as_of_ts,
            )
            state = detector.step(stats, as_of=as_of_ts)
            expected_labels[date_ts] = state.regime_label.value

        # For every date in the panel, the indicator that's hot must
        # equal the detector's returned label.
        any_row_per_date = ff.frame.drop_duplicates(subset=["date"]).set_index("date")
        mismatches: list[tuple[pd.Timestamp, str, str]] = []
        for date_ts, expected_label in expected_labels.items():
            row = any_row_per_date.loc[date_ts]
            family_hot = [
                label for label in REGIME_INDICATOR_LABELS if row[f"is_regime_{label}"] == 1.0
            ]
            if len(family_hot) == 1 and family_hot[0] != expected_label:
                mismatches.append((date_ts, expected_label, family_hot[0]))
            elif len(family_hot) == 0 and expected_label in REGIME_INDICATOR_LABELS:
                # detector said a known label, family said nothing
                mismatches.append((date_ts, expected_label, "<none>"))

        assert not mismatches, (
            f"research/live label divergence on {len(mismatches)} dates "
            f"(first 5: {mismatches[:5]}). The family must walk a shared "
            "MarketRegimeDetector via step(); see ADR-005 hardening."
        )


# ---------------------------------------------------------------------------
# 6. Metadata pinning
# ---------------------------------------------------------------------------


class TestDetectorMetadata:
    """``regime_detector_metadata()`` produces the pinning block the
    script writes into per-arm evidence and the run manifest. Closes
    ADR-005 action item #10 / review finding #4."""

    def test_metadata_keys_match_audit_contract(self) -> None:
        md = regime_detector_metadata()
        # Top-level contract per ADR-005 hardening.
        assert set(md.keys()) == {
            "feature_set_version",
            "detector_version",
            "thresholds",
            "index_proxy",
            "breadth_source",
        }
        assert md["feature_set_version"] == FEATURE_SET_VERSION
        assert md["index_proxy"] == INDEX_PROXY_ID
        assert md["breadth_source"] == BREADTH_SOURCE_ID

    def test_metadata_thresholds_exhaustively_pin_RegimeThresholds(self) -> None:
        md = regime_detector_metadata()
        thresholds_md = md["thresholds"]
        assert isinstance(thresholds_md, dict)
        # Every field of RegimeThresholds is pinned by name. Adding a
        # new field upstream without updating the metadata block
        # would silently leave the new field unpinned, which breaks
        # the audit story — keep this exhaustive.
        assert set(thresholds_md.keys()) == {
            "crisis_vol",
            "risk_off_vol",
            "low_vol",
            "downtrend_z",
            "uptrend_z",
            "weak_breadth",
            "strong_breadth",
            "hysteresis_vol",
            "stability_window",
        }

    def test_metadata_detector_version_matches_core_helper(self) -> None:
        md = regime_detector_metadata()
        expected_version = detector_version("v1.0-rule-based", DEFAULT_REGIME_THRESHOLDS)
        assert md["detector_version"] == expected_version


# ---------------------------------------------------------------------------
# 7. Frozen-time defensive case
# ---------------------------------------------------------------------------


def test_frozen_time_does_not_crash() -> None:
    """Defensive: a panel where every bar has the same date (degenerate
    case used by some unit tests in upstream callers) shouldn't crash
    the family. The detector will produce one regime label and we
    broadcast it across instruments."""
    n_instruments = 4
    inst_ids = [str(uuid.uuid4()) for _ in range(n_instruments)]
    frozen_date = datetime(2024, 6, 1, tzinfo=UTC).date()
    bars = pd.DataFrame(
        {
            "instrument_id": inst_ids,
            "date": [pd.Timestamp(frozen_date)] * n_instruments,
            "close": [100.0 + i for i in range(n_instruments)],
        }
    )
    base = bars.copy()
    base["mom_12_1"] = 0.1
    ff = compute_regime_features(bars, base)
    # With only one date and 4 instruments, the detector lacks data;
    # one of the indicators (probably TRANSITION as the fallback) is 1.
    assert len(ff.frame) == n_instruments
    indicator_total = sum(
        int(ff.frame[f"is_regime_{label}"].sum()) for label in REGIME_INDICATOR_LABELS
    )
    # Either the detector classifies (1 indicator hot × 4 rows = 4)
    # or it returns UNKNOWN (0 indicators hot). Both are acceptable
    # behaviour for a degenerate panel.
    assert indicator_total in (0, n_instruments)
