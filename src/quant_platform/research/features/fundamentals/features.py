"""``fundamentals-plus-v1`` feature factory.

Expands the legacy 9-feature Sharadar starter into a ~40-feature
fundamentals catalog covering quality, value, growth, fundamental
acceleration, cash-flow, leverage, and capital-allocation factors. All
features are computed at the ``(instrument_id, datekey)`` grain and
follow the standard point-in-time discipline already documented in
:mod:`quant_platform.research.fundamentals.sharadar`: ``datekey`` is the
SEC filing date — the moment the row became publicly knowable — and
**not** ``calendardate`` (the fiscal period end).

What's implemented vs. brief
----------------------------

The brief lists ~50–100 candidate fundamentals features. The subset
implemented here is exactly the set that the current Sharadar SF1 ARQ
projection actually supports. The following brief features were
intentionally **not** implemented in this version, with the data each
would need to land:

* ``ebitda_margin``, ``net_debt_to_ebitda`` — need depreciation &
  amortization to derive EBITDA. SF1 ARQ doesn't expose D&A.
* ``interest_coverage`` — needs interest expense (not in current
  projection).
* ``current_ratio``, ``quick_ratio``, ``working_capital_to_assets`` —
  need the current-asset / current-liability split (we only have total
  assets / total liabilities) and inventory.
* ``rd_to_sales``, ``sgna_to_sales`` — need R&D and SG&A broken out of
  ``opex``.
* ``buyback_yield``, ``shareholder_yield`` — need explicit repurchase
  amounts. The closest proxy here is ``low_share_issuance_yoy``
  (share-count growth, inverted).

Each missing feature is intentionally absent rather than approximated;
when the SF1 projection grows, add the spec + compute and bump
``FEATURE_SET_VERSION``.

Direction-aware exports
-----------------------

The platform-wide contract is unchanged from the legacy 9-feature
starter: every exported feature is positive-oriented unless the
:class:`FeatureSpec` says otherwise. Negative-premium quantities are
inverted at the family level and renamed (``low_debt_to_equity``,
``low_asset_growth_yoy``, ``low_share_issuance_yoy``, ...). Features
whose direction in the cross-section is empirically uncertain
(``revenue_growth_qoq`` — short-horizon growth has weak cross-sectional
IC; ``equity_multiplier`` — interpreted as both leverage proxy and
profitability multiplier) ship with
``expected_direction="unknown"`` and ``larger_is_better=False``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quant_platform.research.features.contracts import FeatureFrame, FeatureSpec
from quant_platform.research.features.fundamentals.config import (
    DEFAULT_CONFIG,
    FundamentalsConfig,
)
from quant_platform.research.features.fundamentals.panel import (
    prepare_fundamentals_panel,
)
from quant_platform.research.features.transforms import (
    CALENDAR_DAYS_PER_QUARTER,
    KEY_COLUMNS_FUNDAMENTALS,
    coerce_numeric,
    group_by_instrument,
    ones_like,
    safe_div,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from quant_platform.research.fundamentals.sharadar import SharadarPanel


# ---------------------------------------------------------------------------
# Feature catalog
# ---------------------------------------------------------------------------


def _spec(
    name: str,
    *,
    description: str,
    expected_direction: str,
    required_inputs: tuple[str, ...],
    lookback_quarters: int,
    version: str,
    larger_is_better: bool = True,
    canonical_name: str | None = None,
    aliases: tuple[str, ...] = (),
) -> FeatureSpec:
    """Construct a fundamentals FeatureSpec with the family defaults.

    ``lookback_quarters`` is converted to a calendar-day estimate
    (~91d per quarter) for the ``lookback_days`` field; the precise
    boundary is policed by the panel preparator via ``min_periods``,
    not by this estimate. ``signal_timestamp`` is left at the default
    ``"eod_after_close"`` — fundamentals are filed mid-day but
    consumed in next-day rebalances, which the EOD timestamp matches.
    """
    return FeatureSpec(
        name=name,
        family="fundamentals",
        description=description,
        expected_direction=expected_direction,  # type: ignore[arg-type]
        required_inputs=required_inputs,
        point_in_time=True,
        lookback_days=lookback_quarters * CALENDAR_DAYS_PER_QUARTER,
        version=version,
        larger_is_better=larger_is_better,
        canonical_name=canonical_name,
        aliases=aliases,
    )


def _build_specs(version: str) -> tuple[FeatureSpec, ...]:
    specs: list[FeatureSpec] = []

    # ---- Quality / Profitability ----
    specs.extend(
        [
            _spec(
                "roe_ttm",
                description=(
                    "Trailing 4Q return on equity: netinc_ttm / equity_4q_avg. "
                    "Larger = stronger profitability per unit of equity."
                ),
                expected_direction="+",
                required_inputs=("netinc", "equity"),
                lookback_quarters=4,
                version=version,
            ),
            _spec(
                "roa_ttm",
                description=(
                    "Trailing 4Q return on assets: netinc_ttm / assets_4q_avg. "
                    "Asset-base-adjusted profitability."
                ),
                expected_direction="+",
                required_inputs=("netinc", "assets"),
                lookback_quarters=4,
                version=version,
            ),
            _spec(
                "roic_ttm",
                description=(
                    "Return on invested capital proxy: netinc_ttm / "
                    "(debt + equity - cashneq) (4Q-avg invested capital). "
                    "Approximation — true ROIC uses NOPAT and would need a "
                    "tax rate plus EBIT decomposition that the SF1 ARQ "
                    "projection does not expose."
                ),
                expected_direction="+",
                required_inputs=("netinc", "debt", "equity", "cashneq"),
                lookback_quarters=4,
                version=version,
            ),
            _spec(
                "gross_margin",
                description="gp / revenue (current quarter).",
                expected_direction="+",
                required_inputs=("gp", "revenue"),
                lookback_quarters=1,
                version=version,
            ),
            _spec(
                "operating_margin",
                description=(
                    "(gp - opex) / revenue (current quarter). Operating "
                    "income is derived from gp - opex because SF1 ARQ does "
                    "not ship a standalone opinc column."
                ),
                expected_direction="+",
                required_inputs=("gp", "opex", "revenue"),
                lookback_quarters=1,
                version=version,
            ),
            _spec(
                "net_margin",
                description="netinc / revenue (current quarter).",
                expected_direction="+",
                required_inputs=("netinc", "revenue"),
                lookback_quarters=1,
                version=version,
            ),
            _spec(
                "asset_turnover",
                description="revenue / assets (current quarter).",
                expected_direction="+",
                required_inputs=("revenue", "assets"),
                lookback_quarters=1,
                version=version,
            ),
            _spec(
                "equity_multiplier",
                description=(
                    "assets / equity. Interpreted as a leverage proxy "
                    "(higher = more leveraged) AND a DuPont multiplier "
                    "(higher = larger asset base per equity dollar). The "
                    "direction in the cross-section depends on the regime, "
                    "so this feature ships with expected_direction='unknown'."
                ),
                expected_direction="unknown",
                required_inputs=("assets", "equity"),
                lookback_quarters=1,
                version=version,
                larger_is_better=False,
            ),
            _spec(
                "gross_profitability_q",
                description=(
                    "gp / assets (current quarter). Novy-Marx (2013) gross profitability factor."
                ),
                expected_direction="+",
                required_inputs=("gp", "assets"),
                lookback_quarters=1,
                version=version,
            ),
            _spec(
                "cash_to_assets",
                description="cashneq / assets. Balance-sheet liquidity proxy.",
                expected_direction="+",
                required_inputs=("cashneq", "assets"),
                lookback_quarters=1,
                version=version,
            ),
        ]
    )

    # ---- Growth ----
    specs.extend(
        [
            _spec(
                "revenue_growth_yoy",
                description=(
                    "revenue_ttm / revenue_ttm.shift(4) - 1. TTM-based to "
                    "smooth fiscal-quarter seasonality."
                ),
                expected_direction="+",
                required_inputs=("revenue",),
                lookback_quarters=8,
                version=version,
            ),
            _spec(
                "revenue_growth_qoq",
                description=(
                    "revenue / revenue.shift(1) - 1. Short-horizon growth; "
                    "noisy in the cross-section because of fiscal-quarter "
                    "calendar effects, so ships as unknown-direction."
                ),
                expected_direction="unknown",
                required_inputs=("revenue",),
                lookback_quarters=2,
                version=version,
                larger_is_better=False,
            ),
            _spec(
                "gross_profit_growth_yoy",
                description="gp_ttm / gp_ttm.shift(4) - 1.",
                expected_direction="+",
                required_inputs=("gp",),
                lookback_quarters=8,
                version=version,
            ),
            _spec(
                "operating_income_growth_yoy",
                description="opinc_ttm / opinc_ttm.shift(4) - 1.",
                expected_direction="+",
                required_inputs=("gp", "opex"),
                lookback_quarters=8,
                version=version,
            ),
            _spec(
                "net_income_growth_yoy",
                description="netinc_ttm / netinc_ttm.shift(4) - 1.",
                expected_direction="+",
                required_inputs=("netinc",),
                lookback_quarters=8,
                version=version,
            ),
            _spec(
                "fcf_growth_yoy",
                description="fcf_ttm / fcf_ttm.shift(4) - 1.",
                expected_direction="+",
                required_inputs=("fcf",),
                lookback_quarters=8,
                version=version,
            ),
            _spec(
                "equity_growth_yoy",
                description="equity / equity.shift(4) - 1.",
                expected_direction="+",
                required_inputs=("equity",),
                lookback_quarters=5,
                version=version,
            ),
            _spec(
                "low_asset_growth_yoy",
                description=(
                    "Sign-flipped YoY asset growth: -(assets / assets.shift(4) - 1). "
                    "Cooper-Gulen-Schill (2008): slower balance-sheet expansion "
                    "predicts higher returns."
                ),
                expected_direction="+",
                required_inputs=("assets",),
                lookback_quarters=5,
                version=version,
            ),
        ]
    )

    # ---- Fundamental acceleration ----
    specs.extend(
        [
            _spec(
                "revenue_growth_accel",
                description=(
                    "revenue_growth_yoy - revenue_growth_yoy.shift(4). The "
                    "market often rewards improvement, not static cheapness."
                ),
                expected_direction="+",
                required_inputs=("revenue",),
                lookback_quarters=12,
                version=version,
            ),
            _spec(
                "gross_margin_delta",
                description="gross_margin - gross_margin.shift(4).",
                expected_direction="+",
                required_inputs=("gp", "revenue"),
                lookback_quarters=5,
                version=version,
            ),
            _spec(
                "operating_margin_delta",
                description="operating_margin - operating_margin.shift(4).",
                expected_direction="+",
                required_inputs=("gp", "opex", "revenue"),
                lookback_quarters=5,
                version=version,
            ),
            _spec(
                "net_margin_delta",
                description="net_margin - net_margin.shift(4).",
                expected_direction="+",
                required_inputs=("netinc", "revenue"),
                lookback_quarters=5,
                version=version,
            ),
            _spec(
                "roe_delta_yoy",
                description="roe_ttm - roe_ttm.shift(4).",
                expected_direction="+",
                required_inputs=("netinc", "equity"),
                lookback_quarters=8,
                version=version,
            ),
            _spec(
                "roa_delta_yoy",
                description="roa_ttm - roa_ttm.shift(4).",
                expected_direction="+",
                required_inputs=("netinc", "assets"),
                lookback_quarters=8,
                version=version,
            ),
            _spec(
                "fcf_yield_delta",
                description="fcf_yield_ttm - fcf_yield_ttm.shift(4).",
                expected_direction="+",
                required_inputs=("fcf", "marketcap"),
                lookback_quarters=8,
                version=version,
            ),
        ]
    )

    # ---- Cash flow ----
    specs.extend(
        [
            _spec(
                "fcf_margin",
                description="fcf_ttm / revenue_ttm.",
                expected_direction="+",
                required_inputs=("fcf", "revenue"),
                lookback_quarters=4,
                version=version,
            ),
            _spec(
                "cfo_margin",
                description="ncfo_ttm / revenue_ttm.",
                expected_direction="+",
                required_inputs=("ncfo", "revenue"),
                lookback_quarters=4,
                version=version,
            ),
            _spec(
                "capex_to_assets",
                description=(
                    "abs(capex_ttm) / assets_4q_avg. Sharadar stores capex as a "
                    "negative cash outflow; absolute-value normalises the "
                    "intensity. Direction is unknown — high capex intensity is "
                    "growth-positive in some regimes and balance-sheet-burdening "
                    "in others."
                ),
                expected_direction="unknown",
                required_inputs=("capex", "assets"),
                lookback_quarters=4,
                version=version,
                larger_is_better=False,
            ),
            _spec(
                "capex_to_sales",
                description="abs(capex_ttm) / revenue_ttm.",
                expected_direction="unknown",
                required_inputs=("capex", "revenue"),
                lookback_quarters=4,
                version=version,
                larger_is_better=False,
            ),
            _spec(
                "cash_conversion",
                description=(
                    "ncfo_ttm / netinc_ttm. Larger = more of reported earnings "
                    "are backed by operating cash flow."
                ),
                expected_direction="+",
                required_inputs=("ncfo", "netinc"),
                lookback_quarters=4,
                version=version,
                aliases=("cfo_to_net_income",),
            ),
            _spec(
                "cfo_to_net_income",
                description=(
                    "Alias of cash_conversion (same formula, second name from "
                    "the brief). The canonical column is cash_conversion."
                ),
                expected_direction="+",
                required_inputs=("ncfo", "netinc"),
                lookback_quarters=4,
                version=version,
                canonical_name="cash_conversion",
            ),
            _spec(
                "fcf_to_net_income",
                description="fcf_ttm / netinc_ttm.",
                expected_direction="+",
                required_inputs=("fcf", "netinc"),
                lookback_quarters=4,
                version=version,
            ),
            _spec(
                "low_accruals_4q",
                description=(
                    "Sign-flipped trailing 4Q accruals: "
                    "-((netinc_ttm - ncfo_ttm) / assets_4q_avg). Sloan (1996). "
                    "Larger = cleaner, more cash-backed earnings."
                ),
                expected_direction="+",
                required_inputs=("netinc", "ncfo", "assets"),
                lookback_quarters=4,
                version=version,
            ),
        ]
    )

    # ---- Value ----
    specs.extend(
        [
            _spec(
                "book_to_price",
                description="1 / pb. Reciprocal so larger = cheaper.",
                expected_direction="+",
                required_inputs=("pb",),
                lookback_quarters=1,
                version=version,
            ),
            _spec(
                "earnings_to_price",
                description="1 / pe. Reciprocal so larger = cheaper.",
                expected_direction="+",
                required_inputs=("pe",),
                lookback_quarters=1,
                version=version,
            ),
            _spec(
                "fcf_yield_ttm",
                description="fcf_ttm / marketcap. Cash-flow-based valuation.",
                expected_direction="+",
                required_inputs=("fcf", "marketcap"),
                lookback_quarters=4,
                version=version,
            ),
        ]
    )

    # ---- Leverage ----
    specs.extend(
        [
            _spec(
                "low_debt_to_equity",
                description="-(debt / equity). Larger = lower leverage.",
                expected_direction="+",
                required_inputs=("debt", "equity"),
                lookback_quarters=1,
                version=version,
            ),
            _spec(
                "low_debt_to_assets",
                description="-(debt / assets). Asset-base leverage, inverted.",
                expected_direction="+",
                required_inputs=("debt", "assets"),
                lookback_quarters=1,
                version=version,
            ),
            _spec(
                "low_net_debt_to_marketcap",
                description=(
                    "-((debt - cashneq) / marketcap). Substitute for "
                    "net-debt-to-EBITDA — EBITDA is not derivable from the "
                    "current SF1 projection. Larger = lower net leverage "
                    "scaled by market value."
                ),
                expected_direction="+",
                required_inputs=("debt", "cashneq", "marketcap"),
                lookback_quarters=1,
                version=version,
            ),
        ]
    )

    # ---- Capital allocation ----
    specs.extend(
        [
            _spec(
                "dividend_yield",
                description=("Sharadar's pre-computed divyield (decimal: 0.025 = 2.5%)."),
                expected_direction="+",
                required_inputs=("divyield",),
                lookback_quarters=1,
                version=version,
            ),
            _spec(
                "low_share_issuance_yoy",
                description=(
                    "-(sharesbas / sharesbas.shift(4) - 1). Larger = fewer "
                    "new shares issued. Proxies the buyback-yield direction "
                    "without explicit repurchase data."
                ),
                expected_direction="+",
                required_inputs=("sharesbas",),
                lookback_quarters=5,
                version=version,
            ),
        ]
    )

    return tuple(specs)


FEATURE_SPECS: tuple[FeatureSpec, ...] = _build_specs(DEFAULT_CONFIG.version)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
DEFAULT_TRAINING_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.is_alias
)
_SPEC_BY_NAME: Mapping[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SPECS}


def _specs_for_config(config: FundamentalsConfig) -> tuple[FeatureSpec, ...]:
    """Return specs tagged with ``config.version``.

    For the production config the module-level :data:`FEATURE_SPECS`
    is returned unchanged; for a custom version (e.g. a test that
    wants to round-trip an experimental tag without colliding with
    the production registry entry), specs are re-built with the new
    version pinned. Mirrors the helper of the same name in
    :mod:`quant_platform.research.features.price_volume.features` so
    both families honour ``config.version`` symmetrically.
    """
    if config.version == DEFAULT_CONFIG.version:
        return FEATURE_SPECS
    return _build_specs(config.version)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def _compute_quality_block(df: pd.DataFrame) -> None:
    """Fill the quality / profitability columns on ``df`` in place."""
    df["roe_ttm"] = safe_div(df["netinc_ttm"], df["equity_4q_avg"])
    df["roa_ttm"] = safe_div(df["netinc_ttm"], df["assets_4q_avg"])

    invested_capital = df["debt"] + df["equity"] - df["cashneq"]
    # The proxy is unsigned on the denominator; firms with negative
    # invested capital (cash-rich net-debt-zero balance sheets) are
    # economically uninterpretable here, so they map to NaN.
    df["roic_ttm"] = safe_div(df["netinc_ttm"], invested_capital)

    df["gross_margin"] = safe_div(df["gp"], df["revenue"])
    df["operating_margin"] = safe_div(df["opinc"], df["revenue"])
    df["net_margin"] = safe_div(df["netinc"], df["revenue"])
    df["asset_turnover"] = safe_div(df["revenue"], df["assets"])
    df["equity_multiplier"] = safe_div(df["assets"], df["equity"])
    df["gross_profitability_q"] = safe_div(df["gp"], df["assets"])
    df["cash_to_assets"] = safe_div(df["cashneq"], df["assets"])


def _compute_growth_block(df: pd.DataFrame) -> None:
    df["revenue_growth_yoy"] = safe_div(df["revenue_ttm"], df["revenue_ttm_lag4"]) - 1.0
    df["revenue_growth_qoq"] = safe_div(df["revenue"], df["revenue_lag1"]) - 1.0
    df["gross_profit_growth_yoy"] = safe_div(df["gp_ttm"], df["gp_ttm_lag4"]) - 1.0
    df["operating_income_growth_yoy"] = safe_div(df["opinc_ttm"], df["opinc_ttm_lag4"]) - 1.0
    df["net_income_growth_yoy"] = safe_div(df["netinc_ttm"], df["netinc_ttm_lag4"]) - 1.0
    df["fcf_growth_yoy"] = safe_div(df["fcf_ttm"], df["fcf_ttm_lag4"]) - 1.0
    df["equity_growth_yoy"] = safe_div(df["equity"], df["equity_lag4"]) - 1.0

    raw_asset_growth = safe_div(df["assets"], df["assets_lag4"]) - 1.0
    df["low_asset_growth_yoy"] = -raw_asset_growth


def _compute_acceleration_block(df: pd.DataFrame) -> None:
    # Materialise fcf_yield once so the acceleration column re-uses the
    # same series as the value-block ``fcf_yield_ttm``. Pandas GroupBy
    # holds a *name-based* view of df, so referencing the new column via
    # the existing grouped handle works only after a fresh ``groupby``.
    df["fcf_yield_ttm"] = safe_div(df["fcf_ttm"], df["marketcap"])
    grouped = group_by_instrument(df)

    df["revenue_growth_accel"] = df["revenue_growth_yoy"] - grouped["revenue_growth_yoy"].shift(4)
    df["gross_margin_delta"] = df["gross_margin"] - grouped["gross_margin"].shift(4)
    df["operating_margin_delta"] = df["operating_margin"] - grouped["operating_margin"].shift(4)
    df["net_margin_delta"] = df["net_margin"] - grouped["net_margin"].shift(4)
    df["roe_delta_yoy"] = df["roe_ttm"] - grouped["roe_ttm"].shift(4)
    df["roa_delta_yoy"] = df["roa_ttm"] - grouped["roa_ttm"].shift(4)
    df["fcf_yield_delta"] = df["fcf_yield_ttm"] - grouped["fcf_yield_ttm"].shift(4)


def _compute_cashflow_block(df: pd.DataFrame) -> None:
    df["fcf_margin"] = safe_div(df["fcf_ttm"], df["revenue_ttm"])
    df["cfo_margin"] = safe_div(df["ncfo_ttm"], df["revenue_ttm"])
    df["capex_to_assets"] = safe_div(df["capex_ttm"].abs(), df["assets_4q_avg"])
    df["capex_to_sales"] = safe_div(df["capex_ttm"].abs(), df["revenue_ttm"])
    df["cash_conversion"] = safe_div(df["ncfo_ttm"], df["netinc_ttm"])
    df["cfo_to_net_income"] = df["cash_conversion"]  # alias of cash_conversion
    df["fcf_to_net_income"] = safe_div(df["fcf_ttm"], df["netinc_ttm"])

    accruals_ttm = df["netinc_ttm"] - df["ncfo_ttm"]
    df["low_accruals_4q"] = -safe_div(accruals_ttm, df["assets_4q_avg"])


def _compute_value_block(df: pd.DataFrame) -> None:
    df["book_to_price"] = safe_div(ones_like(df["pb"]), df["pb"])
    df["earnings_to_price"] = safe_div(ones_like(df["pe"]), df["pe"])
    # ``fcf_yield_ttm`` is already populated by _compute_acceleration_block
    # (which uses it as the base series for fcf_yield_delta); avoid the
    # duplicate computation here so the two columns stay byte-identical.


def _compute_leverage_block(df: pd.DataFrame) -> None:
    df["low_debt_to_equity"] = -safe_div(df["debt"], df["equity"])
    df["low_debt_to_assets"] = -safe_div(df["debt"], df["assets"])
    net_debt = df["debt"] - df["cashneq"]
    df["low_net_debt_to_marketcap"] = -safe_div(net_debt, df["marketcap"])


def _compute_capital_block(df: pd.DataFrame) -> None:
    # Sharadar divyield is already a per-quarter dividend yield in
    # decimal form; pass through as-is. Non-numeric sentinels (rare)
    # become NaN.
    df["dividend_yield"] = coerce_numeric(df["divyield"])

    raw_issuance = safe_div(df["sharesbas"], df["sharesbas_lag4"]) - 1.0
    df["low_share_issuance_yoy"] = -raw_issuance


def compute_fundamentals_features(
    panel: SharadarPanel,
    *,
    config: FundamentalsConfig = DEFAULT_CONFIG,
) -> FeatureFrame:
    """Compute the ``fundamentals-plus-v1`` panel.

    Neutralisation has moved out of this function. To get the
    sector-median-neutralised view, compose with
    :func:`quant_platform.research.features.neutralization.neutralize_feature_frame`
    on the returned :class:`FeatureFrame`. The legacy
    ``research/fundamentals/features.py:compute_starter_features``
    preserves its old ``sector_neutralize=True`` kwarg by composing
    internally.

    Parameters
    ----------
    panel:
        Loaded :class:`SharadarPanel` (see
        :func:`quant_platform.research.fundamentals.sharadar.load_sharadar_sf1_panel`).
    config:
        :class:`FundamentalsConfig`. Defaults to :data:`DEFAULT_CONFIG`.

    Returns
    -------
    FeatureFrame
        Long-format frame keyed by ``(instrument_id, datekey)`` plus
        one column per spec in :data:`FEATURE_NAMES`, with a populated
        ``feature_specs`` mapping and a per-feature ``coverage`` dict.
    """
    # Honour ``config.version``: a non-default version produces specs
    # tagged with that version, not the module-level default.
    specs = _specs_for_config(config)
    feature_names = tuple(spec.name for spec in specs)
    spec_by_name: dict[str, FeatureSpec] = {spec.name: spec for spec in specs}

    df = prepare_fundamentals_panel(panel, config=config)

    _compute_quality_block(df)
    _compute_growth_block(df)
    # NB: acceleration depends on growth + quality columns being present.
    _compute_acceleration_block(df)
    _compute_cashflow_block(df)
    _compute_value_block(df)
    _compute_leverage_block(df)
    _compute_capital_block(df)

    feature_columns = list(feature_names)

    # Carry-through columns the legacy walk-forward path expects on its
    # fundamentals frame (ticker + calendardate). Missing carry-throughs
    # are tolerated so the new compute remains usable on stripped panels.
    carry_through = [col for col in ("ticker", "calendardate") if col in df.columns]
    output = df[["instrument_id", "datekey", *carry_through, *feature_columns]].copy()
    output = output.replace([np.inf, -np.inf], np.nan)

    coverage = {name: int(output[name].notna().sum()) for name in feature_columns}

    return FeatureFrame(
        frame=output,
        feature_names=feature_names,
        feature_specs=spec_by_name,
        coverage=coverage,
        key_columns=KEY_COLUMNS_FUNDAMENTALS,
    )


__all__ = [
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "compute_fundamentals_features",
]
