"""Panel preparation utilities for the fundamentals-plus feature factory.

The Sharadar SF1 ARQ panel arrives at this layer already de-duplicated
on ``(instrument_id, datekey)`` and sorted by datekey within each
instrument (see :mod:`quant_platform.research.fundamentals.sharadar`).
What it does **not** have is the set of rolling aggregates every
fundamentals feature depends on:

* TTM sums (``netinc_ttm``, ``ncfo_ttm``, ``fcf_ttm``, ``revenue_ttm``,
  ``gp_ttm``, ``opex_ttm``, ``capex_ttm``)
* 4-quarter rolling averages (``equity_4q_avg``, ``assets_4q_avg``,
  ``marketcap_4q_avg``)
* 4-quarter lags (``revenue_lag4``, ``netinc_lag4``, ...) and 1-quarter
  lags for QoQ growth
* Per-quarter derived ``opinc`` = ``gp - opex`` so the operating-margin
  family has a clean numerator

Doing this work once in :func:`prepare_fundamentals_panel` keeps the
feature compute functions short and readable, and guarantees every
feature shares the same TTM / lag conventions.

The function is pure: it consumes a ``SharadarPanel`` and returns a new
DataFrame (with the same row count and ``(instrument_id, datekey)``
ordering) augmented with the helper columns. The input frame is not
mutated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.research.features.fundamentals.config import (
    DEFAULT_CONFIG,
    QOQ_LAG_QUARTERS,
    TTM_WINDOW_QUARTERS,
    YOY_LAG_QUARTERS,
    FundamentalsConfig,
)
from quant_platform.research.features.transforms import (
    group_by_instrument,
    group_rolling_mean,
    group_rolling_sum,
    group_shift,
)

if TYPE_CHECKING:
    import pandas as pd

    from quant_platform.research.features.transforms import MinPeriodsPolicy
    from quant_platform.research.fundamentals.sharadar import SharadarPanel


# Raw Sharadar SF1 columns the legacy 9-feature catalog needs. These
# are non-negotiable: missing here means the panel cannot produce a
# legitimate output and we fail loudly. ``equityavg`` is intentionally
# absent — Sharadar stores it as an object dtype with sentinel strings
# on some rows; we compute our own ``equity_4q_avg`` from ``equity`` to
# keep the dtype clean and the formula auditable.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "datekey",
    "netinc",
    "ncfo",
    "fcf",
    "equity",
    "assets",
    "gp",
    "marketcap",
    "cashneq",
    "debt",
    "pb",
    "pe",
)

#: Deprecated alias kept so old callers do not break. New code should
#: read ``REQUIRED_INPUT_COLUMNS``; this alias will be removed at the
#: next feature-set version bump.
REQUIRED_RAW_COLUMNS: tuple[str, ...] = REQUIRED_INPUT_COLUMNS

# Raw columns the *new* fundamentals-plus features need. Missing here
# is tolerated — the panel preparator synthesises an all-NaN column so
# downstream features that depend on it produce NaN coverage rather
# than raising. Lets the legacy-shaped test fixtures (which only carry
# the columns above) continue to work after migration to v1, and lets
# operators trim the SF1 projection if they don't care about a subset
# of the new features.
OPTIONAL_RAW_COLUMNS: tuple[str, ...] = (
    "revenue",
    "opex",
    "liabilities",
    "capex",
    "divyield",
    "sharesbas",
)


def _validate_inputs(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"prepare_fundamentals_panel: input panel missing required columns: {missing!r}"
        )


def prepare_fundamentals_panel(
    panel: SharadarPanel,
    *,
    config: FundamentalsConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Return a sorted, helper-columns-attached working frame.

    Parameters
    ----------
    panel:
        Loaded :class:`SharadarPanel` (see
        :func:`quant_platform.research.fundamentals.sharadar.load_sharadar_sf1_panel`).
    config:
        :class:`FundamentalsConfig` controlling whether warm-up rows
        emit partial TTM / YoY values. Defaults to the production
        config (``require_full_ttm=True`` / ``require_full_yoy=True``).

    Returns
    -------
    pd.DataFrame
        Long-format frame keyed by ``(instrument_id, datekey)`` with the
        original raw columns plus the helper columns documented in this
        module's docstring.
    """
    import numpy as np  # local import: only needed for the optional-column path

    _validate_inputs(panel.frame)

    df = panel.frame.copy()
    # Synthesize all-NaN columns for any optional inputs the panel does
    # not carry. The legacy synthetic test panels intentionally trim
    # the projection to the columns the legacy 9 features needed; the
    # new features that depend on the missing columns will produce
    # NaN coverage, which is the correct behaviour.
    for col in OPTIONAL_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values(["instrument_id", "datekey"]).reset_index(drop=True)

    policy: MinPeriodsPolicy = "full" if config.require_full_ttm else "partial"

    grouped = group_by_instrument(df)

    # ---- TTM sums (rolling 4-quarter sum) ----
    for raw, ttm_name in (
        ("netinc", "netinc_ttm"),
        ("ncfo", "ncfo_ttm"),
        ("fcf", "fcf_ttm"),
        ("revenue", "revenue_ttm"),
        ("gp", "gp_ttm"),
        ("opex", "opex_ttm"),
        ("capex", "capex_ttm"),
    ):
        df[ttm_name] = group_rolling_sum(grouped[raw], TTM_WINDOW_QUARTERS, policy=policy)

    # ---- 4-quarter rolling averages ----
    for raw, avg_name in (
        ("equity", "equity_4q_avg"),
        ("assets", "assets_4q_avg"),
        ("marketcap", "marketcap_4q_avg"),
    ):
        df[avg_name] = group_rolling_mean(grouped[raw], TTM_WINDOW_QUARTERS, policy=policy)

    # ---- Per-quarter derived: operating income ----
    # Sharadar SF1 ARQ doesn't ship ``opinc`` as a column; derive once
    # here from ``gp - opex`` so every operating-margin feature shares
    # the same definition. ``opinc_ttm`` needs a fresh groupby because
    # ``opinc`` is added to ``df`` after the original ``grouped`` was
    # constructed.
    df["opinc"] = df["gp"] - df["opex"]
    grouped_with_opinc = group_by_instrument(df)
    df["opinc_ttm"] = group_rolling_sum(
        grouped_with_opinc["opinc"], TTM_WINDOW_QUARTERS, policy=policy
    )

    # ---- YoY lags (shift 4 quarters) ----
    yoy_columns = (
        "revenue",
        "revenue_ttm",
        "gp",
        "gp_ttm",
        "opinc",
        "opinc_ttm",
        "netinc",
        "netinc_ttm",
        "fcf_ttm",
        "assets",
        "equity",
        "sharesbas",
    )
    for col in yoy_columns:
        df[f"{col}_lag4"] = group_shift(grouped_with_opinc[col], YOY_LAG_QUARTERS)

    # ---- QoQ lag (shift 1 quarter) — revenue only for now, that is what the
    # brief calls out under "Growth" / revenue_growth_qoq. ----
    df["revenue_lag1"] = group_shift(grouped_with_opinc["revenue"], QOQ_LAG_QUARTERS)

    return df


__all__ = [
    "REQUIRED_INPUT_COLUMNS",
    "REQUIRED_RAW_COLUMNS",  # deprecated alias
    "prepare_fundamentals_panel",
]
