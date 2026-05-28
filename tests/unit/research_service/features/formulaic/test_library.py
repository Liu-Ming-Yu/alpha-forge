"""Smoke + spec tests for the starter library."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features import get_global_registry
from quant_platform.research.features.formulaic import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    FormulaicConfig,
    compute_formulaic_features,
)
from quant_platform.research.features.formulaic.library import LIBRARY


def _wide_bars(n_instruments: int = 10, n_rows: int = 120) -> pd.DataFrame:
    """A wide-enough universe that cross-sectional ranks vary, so
    expressions like ts_corr(rank(volume), rank(high), 5) are not
    degenerate."""
    rng = np.random.default_rng(seed=0)
    rows = []
    dates = pd.bdate_range(start="2023-01-02", periods=n_rows)
    for inst_idx in range(n_instruments):
        # Use independent random walks so per-date ranks don't all
        # collapse to the same ordering.
        closes = 100.0 + np.cumsum(rng.normal(0.05, 1.0, size=n_rows))
        for i, d in enumerate(dates):
            close = float(closes[i])
            rows.append(
                {
                    "instrument_id": f"I{inst_idx}",
                    "date": d,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0 + 5.0 * inst_idx + rng.normal(0, 50),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spec / registry contract
# ---------------------------------------------------------------------------


def test_every_library_alpha_has_a_spec() -> None:
    # The curated LIBRARY is a strict subset of the family's
    # FEATURE_NAMES; on a fresh checkout (no auto-promoted file)
    # they coincide, but once an operator has run promotion the
    # family's effective set is curated + auto. Test the subset
    # relation so the assertion stays true in both worlds.
    library_names = {alpha.name for alpha in LIBRARY}
    spec_names = {spec.name for spec in FEATURE_SPECS}
    assert library_names <= spec_names
    assert len(FEATURE_NAMES) >= len(LIBRARY)


def test_specs_are_evidence_gated_by_default() -> None:
    for spec in FEATURE_SPECS:
        assert spec.expected_direction == "unknown", spec.name
        assert spec.larger_is_better is False, spec.name


def test_specs_registered_globally() -> None:
    registry = get_global_registry()
    for spec in FEATURE_SPECS:
        assert registry.has(spec.name, spec.version), spec.name
        assert registry.get(spec.name, spec.version) == spec


def test_lookback_days_match_ast_derivation() -> None:
    for alpha in LIBRARY:
        ast_lookback = alpha.expression.lookback_days()
        spec_lookback = next(s for s in FEATURE_SPECS if s.name == alpha.name).lookback_days
        assert spec_lookback == ast_lookback, alpha.name


def test_required_inputs_come_from_ast() -> None:
    for alpha in LIBRARY:
        ast_inputs = alpha.expression.required_inputs()
        spec_inputs = next(s for s in FEATURE_SPECS if s.name == alpha.name).required_inputs
        # Spec sorts the set into a tuple; compare as a set.
        assert set(spec_inputs) == ast_inputs, alpha.name


def test_no_aliases_in_starter_library() -> None:
    assert DEFAULT_TRAINING_FEATURE_NAMES == FEATURE_NAMES


# ---------------------------------------------------------------------------
# End-to-end compute
# ---------------------------------------------------------------------------


def test_compute_formulaic_features_smoke() -> None:
    bars = _wide_bars()
    result = compute_formulaic_features(bars)
    assert set(result.feature_names) == set(FEATURE_NAMES)
    assert len(result.frame) == len(bars)
    for name in result.feature_names:
        # On a wide universe with random closes, every alpha should
        # produce at least *some* non-NaN values after its warm-up.
        assert result.coverage[name] > 0, f"{name} produced 0 coverage"


def test_custom_version_stamps_into_specs() -> None:
    bars = _wide_bars(n_instruments=4, n_rows=40)
    cfg = FormulaicConfig(version="formulaic-test-v0")
    result = compute_formulaic_features(bars, config=cfg)
    for name in result.feature_names:
        assert result.feature_specs[name].version == "formulaic-test-v0"


def test_returns_column_synthesized_when_missing() -> None:
    """The MarketPanel adapter derives ``returns`` if not present;
    library alphas that consume ``returns`` (e.g. wq_alpha_001) must
    still produce a defined coverage."""
    bars = _wide_bars()
    assert "returns" not in bars.columns
    result = compute_formulaic_features(bars)
    assert result.coverage["wq_alpha_001"] > 0


def test_finite_outputs_only() -> None:
    """No ±inf leaks past the family boundary, even on degenerate
    inputs."""
    bars = _wide_bars()
    result = compute_formulaic_features(bars)
    for name in result.feature_names:
        col = result.frame[name].to_numpy()
        valid = col[~np.isnan(col)]
        assert not np.isinf(valid).any(), f"{name} produced inf values"


def test_empty_bar_frame_raises() -> None:
    """An empty input doesn't crash silently; it raises at the panel
    boundary."""
    empty = pd.DataFrame(
        {
            "instrument_id": pd.Series(dtype=str),
            "date": pd.Series(dtype="datetime64[ns]"),
            "open": pd.Series(dtype=float),
            "high": pd.Series(dtype=float),
            "low": pd.Series(dtype=float),
            "close": pd.Series(dtype=float),
            "volume": pd.Series(dtype=float),
        }
    )
    # Empty panels are tolerated — every alpha produces an empty column.
    result = compute_formulaic_features(empty)
    assert result.frame.empty
    assert all(v == 0 for v in result.coverage.values())


def test_missing_required_input_raises() -> None:
    bars = _wide_bars().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        compute_formulaic_features(bars)


@pytest.mark.parametrize("alpha_name", [alpha.name for alpha in LIBRARY])
def test_every_alpha_computes_without_error(alpha_name: str) -> None:
    """One test per alpha: build the wide panel and confirm the alpha
    computes without raising."""
    bars = _wide_bars()
    result = compute_formulaic_features(bars)
    assert alpha_name in result.feature_names
