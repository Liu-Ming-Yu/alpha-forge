"""Conformance test: ``fundamentals-plus-v1`` against the real Sharadar panel.

This test runs end-to-end ``compute_fundamentals_features`` against the
SF1 ARQ parquet that lives under ``data/parquet/research/fundamentals/``
and asserts that every feature in the catalog produces non-zero
coverage on production data. It exists as a checked-in safety net so
that a future change to ``panel.py`` / ``features.py`` that silently
breaks production-data shapes — but happens to pass the synthetic-panel
unit tests — gets caught before it lands.

The test **skips** when the Sharadar parquet is absent (the file is
gitignored under ``/data/parquet/``; CI does not carry it). Run
locally after fetching the data via
``scripts/pull_sharadar_sf1.py``::

    pytest tests/integration/test_fundamentals_sharadar_conformance.py -v

The numerical thresholds below are deliberately loose: they catch
"entire feature collapsed to NaN" regressions, not "feature shifted
by 1%" drift. Tighten in a follow-up if production data evolves to
the point where stable per-feature coverage bounds become possible.
"""

from __future__ import annotations

import pytest

from quant_platform.research.features.fundamentals import (
    FEATURE_NAMES,
    compute_fundamentals_features,
)
from quant_platform.research.fundamentals.sharadar import (
    DEFAULT_SF1_PARQUET,
    load_sector_map,
    load_sharadar_sf1_panel,
)

#: Features that *can* legitimately have zero coverage on a particular
#: vintage of SF1 (e.g. ``divyield`` may be all NaN for the universe
#: subset that lands in the ticker map). The test only enforces the
#: "got at least one non-NaN" invariant on features outside this set.
ZERO_COVERAGE_TOLERATED: frozenset[str] = frozenset()


@pytest.mark.skipif(
    not DEFAULT_SF1_PARQUET.exists(),
    reason=(
        f"Sharadar SF1 parquet not present at {DEFAULT_SF1_PARQUET}; run "
        "``scripts/pull_sharadar_sf1.py`` locally to fetch the bundle "
        "before running this conformance test. CI skips this test by "
        "design (the parquet is gitignored)."
    ),
)
def test_every_feature_has_nonzero_coverage_on_real_panel() -> None:
    panel = load_sharadar_sf1_panel()
    assert panel.frame.shape[0] > 1_000, (
        f"Sharadar panel suspiciously small ({panel.frame.shape[0]} rows); "
        "ticker map or filter may have changed."
    )

    result = compute_fundamentals_features(panel)

    zero_coverage_features = [
        name
        for name in FEATURE_NAMES
        if result.coverage[name] == 0 and name not in ZERO_COVERAGE_TOLERATED
    ]
    assert not zero_coverage_features, (
        "Features collapsed to zero coverage on real Sharadar panel: "
        f"{zero_coverage_features!r}. If this is intentional for the "
        "current vintage (column missing upstream, etc.), add the name "
        "to ZERO_COVERAGE_TOLERATED with a comment."
    )


@pytest.mark.skipif(
    not DEFAULT_SF1_PARQUET.exists(),
    reason="Sharadar SF1 parquet not present; see preceding test for details.",
)
def test_sector_neutralization_preserves_feature_count_on_real_panel() -> None:
    """The sector-neutralized variant must produce the same feature
    columns with non-zero coverage as the raw variant."""
    from quant_platform.research.features.neutralization import (
        neutralize_feature_frame,
    )

    panel = load_sharadar_sf1_panel()
    sector_map = load_sector_map()
    raw = compute_fundamentals_features(panel)
    neutralized = neutralize_feature_frame(
        raw,
        by="sector_median",
        sector_map=sector_map,
    )

    assert neutralized.feature_names == raw.feature_names

    # Sector neutralization should not destroy more than ~20% of coverage
    # for any feature (sector medians are well-defined whenever the
    # sector has ≥1 member at the date).
    for name in FEATURE_NAMES:
        raw_cov = raw.coverage[name]
        neutral_cov = neutralized.coverage[name]
        if raw_cov == 0:
            assert neutral_cov == 0
            continue
        retention = neutral_cov / raw_cov
        assert retention > 0.8, (
            f"Sector-neutralised {name!r} retained only {retention:.1%} "
            f"of raw coverage ({neutral_cov} of {raw_cov} rows). A drop "
            "this large indicates the sector map is missing too many "
            "instruments or the neutraliser is leaking NaNs."
        )
