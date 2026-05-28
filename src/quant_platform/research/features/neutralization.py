"""Cross-sectional neutralization transforms.

A feature value at ``(instrument_id, date)`` carries three kinds of
information mixed together: the alpha signal we actually want, the
common-factor exposure of the instrument (sector, industry, size,
liquidity, volatility, market beta), and idiosyncratic noise. Cross-
sectional neutralization removes the common-factor exposure so the
remaining signal is orthogonal to whatever risk the portfolio
constructor is already controlling for.

Two flavours live here:

1. **Cross-sectional standardizers** — ``cross_sectional_rank`` and
   ``cross_sectional_zscore`` map a raw value to a [0, 1] (or
   zero-mean unit-variance) view *within each date*. They do not
   require any covariate; their purpose is to make magnitudes
   comparable across instruments.
2. **Residualizers** — ``neutralize_by_group`` and
   ``neutralize_by_size`` strip out a categorical (sector / industry)
   or continuous (``log(marketcap)``) factor from the feature value
   *within each date*, leaving the residual that is orthogonal to the
   neutralized exposure.

Every helper here is pure: it takes a long-format DataFrame keyed by
``(instrument_id, date)`` (the date column name is configurable), an
output column name, and the configuration it needs, and returns a new
``Series`` aligned to the input. None of the helpers mutate the input
frame, and none reorder its rows. Group-by-date operations use
``transform`` so the output index is always the input index.

Date column name
----------------

Fundamentals features are keyed by ``datekey`` (SEC filing date);
price-volume features are keyed by ``date`` (trading day). The
neutralization helpers take the date column name as an argument so
the same module serves both families.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from quant_platform.research.features.contracts import FeatureFrame
from quant_platform.research.features.transforms import UNKNOWN_GROUP_SENTINEL

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

NeutralizationKind = Literal["sector_median", "sector_mean", "size_residual"]


def cross_sectional_rank(
    frame: pd.DataFrame,
    column: str,
    *,
    date_column: str = "date",
    pct: bool = True,
) -> pd.Series:
    """Per-date rank of ``column``.

    Parameters
    ----------
    frame:
        Long-format DataFrame with at least ``date_column`` and ``column``.
    column:
        Name of the value column to rank.
    date_column:
        Name of the date/datekey column to group by. Defaults to
        ``"date"`` (price-volume convention); fundamentals callers pass
        ``"datekey"``.
    pct:
        When ``True`` (default), returns ranks in ``[0, 1]`` (pandas
        ``Series.rank(pct=True)``). When ``False``, returns 1-based
        integer ranks per date.

    Notes
    -----
    Uses ``method="average"`` so ties contribute equally; this matches
    standard rank-IC conventions. NaN values stay NaN.
    """
    grouped = frame.groupby(date_column, sort=False, group_keys=False)[column]
    return grouped.transform(lambda s: s.rank(method="average", pct=pct))


def cross_sectional_zscore(
    frame: pd.DataFrame,
    column: str,
    *,
    date_column: str = "date",
    ddof: int = 0,
    winsorize: float | None = None,
) -> pd.Series:
    """Per-date z-score of ``column``.

    Parameters
    ----------
    winsorize:
        If set to ``p`` (``0 < p < 0.5``), clip the input to its
        per-date ``[p, 1 - p]`` quantile range before standardizing.
        This is the standard defence against single-name outliers
        pulling the cross-sectional mean/std. ``None`` (default)
        leaves the values untouched.
    ddof:
        Standard deviation degrees-of-freedom; defaults to 0 (population
        std) to match the historical convention of most cross-sectional
        factor pipelines. Pass ``ddof=1`` for the sample-std variant.

    Notes
    -----
    NaN inputs stay NaN. Dates where the per-date std collapses to zero
    (e.g. constant feature value or a single instrument that day) get
    NaN output rather than ``inf``.
    """
    if winsorize is not None and not (0.0 < winsorize < 0.5):
        raise ValueError("winsorize must lie in the open interval (0, 0.5)")

    grouped = frame.groupby(date_column, sort=False, group_keys=False)[column]

    def _standardize(s: pd.Series) -> pd.Series:
        values = s
        if winsorize is not None:
            lo = values.quantile(winsorize)
            hi = values.quantile(1.0 - winsorize)
            values = values.clip(lower=lo, upper=hi)
        mean = values.mean()
        std = values.std(ddof=ddof)
        if not np.isfinite(std) or std == 0.0:
            # Whole cross-section is constant (or all-NaN) on this date;
            # standardization is undefined — return NaN rather than 0/0.
            return values * np.nan
        return (values - mean) / std

    return grouped.transform(_standardize)


def neutralize_by_group(
    frame: pd.DataFrame,
    column: str,
    *,
    group_column: str,
    date_column: str = "date",
    statistic: str = "median",
) -> pd.Series:
    """Subtract the per-(date, group) statistic from ``column``.

    Parameters
    ----------
    frame:
        Long-format DataFrame containing ``column``, ``group_column``,
        and ``date_column``.
    column:
        Value column to neutralize.
    group_column:
        Categorical column to neutralize against (typically
        ``"sector"`` or ``"industry"``).
    date_column:
        Per-date grouping. Defaults to ``"date"``.
    statistic:
        Either ``"median"`` (default; the brief's recommended choice for
        outlier-robust sectoral neutralization) or ``"mean"``.

    Notes
    -----
    Single-name groups produce a residual of zero for the lone member
    (that name *is* the group), which is the correct semantic: the
    residual against the group's own location is zero. NaN feature
    values remain NaN.

    Rows whose ``group_column`` is null fall through to the global
    per-date statistic, so we never silently drop names with missing
    sector mappings.
    """
    if statistic not in {"median", "mean"}:
        raise ValueError(f"unknown neutralization statistic: {statistic!r}")

    # Use the shared :data:`UNKNOWN_GROUP_SENTINEL` for missing-group rows
    # so they share a pseudo-group rather than each landing in their own
    # group of size 1.
    work = frame[[date_column, group_column, column]].copy()
    work[group_column] = work[group_column].astype("object").fillna(UNKNOWN_GROUP_SENTINEL)

    grouped = work.groupby([date_column, group_column], sort=False, group_keys=False)
    if statistic == "median":
        group_stat = grouped[column].transform("median")
    else:
        group_stat = grouped[column].transform("mean")
    return (work[column] - group_stat).astype(float)


def neutralize_by_size(
    frame: pd.DataFrame,
    column: str,
    *,
    size_column: str = "marketcap",
    date_column: str = "date",
    log_size: bool = True,
) -> pd.Series:
    """Residualize ``column`` against (log) size, per date.

    Implements the per-date univariate OLS
    ``column ~ alpha + beta * size + epsilon`` and returns ``epsilon``,
    so the residual is orthogonal to the size factor by construction.

    Parameters
    ----------
    size_column:
        Column carrying the size proxy. ``"marketcap"`` by default.
    log_size:
        If ``True`` (default), the regressor is ``log(size)`` clipped
        at zero so non-positive market caps map to NaN rather than
        ``-inf``.

    Notes
    -----
    Rows with NaN feature or NaN size are excluded from the per-date
    fit and the residual is NaN for those rows. Dates with fewer than
    two valid observations produce an all-NaN residual for that date
    (a univariate OLS needs at least 2 points).
    """
    import pandas as pd  # local import; only used for the typed buffer below

    work = frame[[date_column, size_column, column]].copy()
    if log_size:
        size = work[size_column].astype(float)
        work["_size"] = np.where(size > 0, np.log(size.where(size > 0)), np.nan)
    else:
        work["_size"] = work[size_column].astype(float)

    # Per-date OLS residual. Build the output buffer index-aligned to
    # ``work`` and fill per group; this avoids the ``DataFrameGroupBy
    # .apply`` indexing edge cases (output index gets prefixed with the
    # groupby key under certain pandas paths, which then needs a
    # reset/reindex that's easy to get wrong).
    residual = pd.Series(np.nan, index=work.index, dtype=float)

    for _, idx in work.groupby(date_column, sort=False).groups.items():
        x = work.loc[idx, "_size"].astype(float)
        y = work.loc[idx, column].astype(float)
        mask = y.notna() & x.notna()
        if int(mask.sum()) < 2:
            continue
        x_valid = x[mask]
        y_valid = y[mask]
        x_mean = float(x_valid.mean())
        y_mean = float(y_valid.mean())
        x_var = float(((x_valid - x_mean) ** 2).sum())
        if x_var == 0.0 or not np.isfinite(x_var):
            # Constant regressor on this date: OLS is degenerate. Fall
            # back to mean-centring the response so the residual is at
            # least zero-mean (no slope to remove).
            residual.loc[x_valid.index] = (y_valid - y_mean).to_numpy()
            continue
        beta = float(((x_valid - x_mean) * (y_valid - y_mean)).sum() / x_var)
        alpha = y_mean - beta * x_mean
        residual.loc[x_valid.index] = (y_valid - (alpha + beta * x_valid)).to_numpy()

    return residual


def attach_group_map(
    frame: pd.DataFrame,
    *,
    group_map: Mapping[str, str],
    instrument_column: str = "instrument_id",
    output_column: str = "sector",
    unknown_fill: str = UNKNOWN_GROUP_SENTINEL,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a categorical group column attached.

    Convenience wrapper around ``frame[instrument_column].map(group_map)``
    that fills missing entries with a sentinel rather than NaN so
    downstream ``groupby`` calls don't drop those rows. Used by the
    fundamentals family to attach ``sector`` from the Sharadar ticker
    map before sector-neutralization.
    """
    out = frame.copy()
    mapped = out[instrument_column].astype(str).map(group_map)
    out[output_column] = mapped.fillna(unknown_fill).astype("object")
    return out


def neutralize_feature_frame(
    feature_frame: FeatureFrame,
    *,
    by: NeutralizationKind,
    sector_map: Mapping[str, str] | None = None,
    size_column: str = "marketcap",
    log_size: bool = True,
) -> FeatureFrame:
    """Return a new :class:`FeatureFrame` with every feature column neutralised.

    Designed to live **outside** the family ``compute_*_features``
    function: families produce raw values, this helper applies the
    cross-sectional residualisation as a post-processing step. The
    one-rule-across-families property is what the previous design
    lacked — ``compute_fundamentals_features`` had a ``sector_neutralize``
    kwarg, ``compute_price_volume_features`` did not, and any new
    family would have had to pick a side.

    Parameters
    ----------
    feature_frame:
        Input :class:`FeatureFrame`. The frame's ``key_columns`` are
        introspected to find the date column (the last entry is taken
        as the time axis; the first is the instrument axis).
    by:
        Which neutralisation to apply:
        * ``"sector_median"`` — per-date sector median subtraction (the
          standard outlier-robust choice; default of the legacy
          ``sector_neutralize=True`` path).
        * ``"sector_mean"`` — same shape but mean-centring; emits
          exactly-zero per-(date, sector) residual mean.
        * ``"size_residual"`` — per-date univariate OLS residual
          against ``log(size_column)``.
    sector_map:
        Required for ``by="sector_*"``. ``instrument_id -> sector``
        mapping (e.g. the dict returned by
        :func:`quant_platform.research.fundamentals.sharadar.load_sector_map`).
    size_column:
        Column to use as the size regressor when ``by="size_residual"``.
        Defaults to ``"marketcap"``.
    log_size:
        Whether to log-transform the size regressor. Default ``True``.

    Returns
    -------
    FeatureFrame
        A new FeatureFrame with the same specs, key columns, and
        coverage **keys** as the input, but with each feature column
        replaced by its neutralised residual. Coverage **counts** are
        recomputed against the residual (rows that lose to the
        regressor become NaN).

    Notes
    -----
    The function does **not** mutate the input frame. The returned
    FeatureFrame is constructed fresh.
    """
    if by in {"sector_median", "sector_mean"} and sector_map is None:
        raise ValueError(f"neutralize_feature_frame: by={by!r} requires sector_map")

    instrument_column, *_, date_column = feature_frame.key_columns
    feature_columns = list(feature_frame.feature_names)
    working = feature_frame.frame.copy()

    if by in {"sector_median", "sector_mean"}:
        statistic = "median" if by == "sector_median" else "mean"
        narrowed_map: Mapping[str, str] = sector_map or {}
        sectors = (
            working[instrument_column].astype(str).map(narrowed_map).fillna(UNKNOWN_GROUP_SENTINEL)
        )
        working["_sector"] = sectors
        for name in feature_columns:
            working[name] = neutralize_by_group(
                working,
                name,
                group_column="_sector",
                date_column=date_column,
                statistic=statistic,
            )
        working = working.drop(columns=["_sector"])
    elif by == "size_residual":
        if size_column not in working.columns:
            raise ValueError(
                f"neutralize_feature_frame: size_column={size_column!r} not "
                f"present on the input frame; columns are {list(working.columns)!r}"
            )
        for name in feature_columns:
            working[name] = neutralize_by_size(
                working,
                name,
                size_column=size_column,
                date_column=date_column,
                log_size=log_size,
            )
    else:  # pragma: no cover — covered by the literal type
        raise ValueError(f"neutralize_feature_frame: unknown kind {by!r}")

    coverage = {name: int(working[name].notna().sum()) for name in feature_columns}

    return FeatureFrame(
        frame=working,
        feature_names=feature_frame.feature_names,
        feature_specs=feature_frame.feature_specs,
        coverage=coverage,
        key_columns=feature_frame.key_columns,
    )


__all__ = [
    "NeutralizationKind",
    "attach_group_map",
    "cross_sectional_rank",
    "cross_sectional_zscore",
    "neutralize_by_group",
    "neutralize_by_size",
    "neutralize_feature_frame",
]
