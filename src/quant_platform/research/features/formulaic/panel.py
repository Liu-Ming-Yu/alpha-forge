"""Input panel adapter for the formulaic alpha factory.

The formulaic family consumes a daily OHLCV bar panel — same shape as
:mod:`~.price_volume.features` consumes — plus optional carry-throughs
(``vwap``, ``marketcap``, ``sector``, ``industry``) for the operators
that need them (``group_rank`` uses ``sector``; some WorldQuant alphas
use ``vwap``). :class:`MarketPanel` is the contract every operator
reads from: it owns a long-format ``DataFrame`` keyed by
``(instrument_id, date)``, validates required columns, and pre-derives
the small handful of columns operators expect by convention
(``returns``, ``dollar_volume``).

Why a wrapper rather than a bare DataFrame? Three reasons:

1. **Validation up front.** Operators read by column name; a typo in a
   library expression should raise at panel construction, not at
   evaluation time per-instrument.
2. **Derived columns once.** ``returns = close.pct_change()`` and
   ``dollar_volume = close * volume`` are computed once in the panel
   and looked up by name; otherwise every operator that needs returns
   would re-derive them.
3. **Caching key.** The :class:`~.evaluator.ExpressionCache` is
   per-panel; bundling the panel into a typed object makes "same data"
   easier to assert than re-hashing a DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quant_platform.research.features.transforms import (
    DEFAULT_KEY_COLUMNS,
    group_by_instrument,
    group_shift,
    safe_div,
)

if TYPE_CHECKING:
    import pandas as pd
    from pandas.core.groupby.generic import DataFrameGroupBy

#: Columns the formulaic family requires on every input panel. Missing
#: any of these is a hard error.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

#: Columns the formulaic family will use if present, but tolerates if
#: absent. Operators that need them (e.g. ``group_rank(close, by="sector")``)
#: fail loudly when the column is required by a specific alpha and missing.
OPTIONAL_INPUT_COLUMNS: tuple[str, ...] = (
    "vwap",
    "marketcap",
    "sector",
    "industry",
)

#: Columns the panel adapter derives if the caller hasn't supplied
#: them. Listed here so library authors can refer to them by name from
#: AST :class:`~.ast.Var` nodes.
DERIVED_COLUMNS: tuple[str, ...] = ("returns", "dollar_volume")


@dataclass(frozen=True)
class MarketPanel:
    """Validated input panel for formulaic-alpha evaluation.

    Attributes
    ----------
    frame:
        Long-format DataFrame sorted by ``(instrument_id, date)``,
        already augmented with ``returns`` and ``dollar_volume`` if
        they weren't already present.
    available_columns:
        Frozen set of every column on :attr:`frame`. Used by the
        evaluator to validate that an expression's ``required_inputs``
        is a subset.
    """

    frame: pd.DataFrame
    available_columns: frozenset[str]

    def grouped(self) -> DataFrameGroupBy:
        """Per-instrument GroupBy view, used by time-series operators."""
        return group_by_instrument(self.frame)

    def has_column(self, name: str) -> bool:
        return name in self.available_columns

    def require_column(self, name: str) -> None:
        if name not in self.available_columns:
            raise KeyError(
                f"MarketPanel does not carry column {name!r}; available: "
                f"{sorted(self.available_columns)!r}"
            )


def build_market_panel(frame: pd.DataFrame) -> MarketPanel:
    """Validate and adapt a raw bar frame into a :class:`MarketPanel`.

    The caller passes a long-format DataFrame keyed by
    ``(instrument_id, date)``. This function:

    1. Rejects frames missing any :data:`REQUIRED_INPUT_COLUMNS`.
    2. Sorts by ``(instrument_id, date)`` and resets the index.
    3. Adds ``returns`` (per-instrument ``close.pct_change``) and
       ``dollar_volume`` (``close * volume``) if they aren't already
       columns. Existing columns by those names are left untouched.

    The returned panel is immutable (frozen dataclass; the inner
    frame is a copy).
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"build_market_panel: input frame missing required columns: {missing!r}; "
            f"got {list(frame.columns)!r}"
        )

    df = frame.copy().sort_values(list(DEFAULT_KEY_COLUMNS)).reset_index(drop=True)

    if "returns" not in df.columns:
        grouped_close = group_by_instrument(df)["close"]
        df["returns"] = safe_div(df["close"], group_shift(grouped_close, 1)) - 1.0
    if "dollar_volume" not in df.columns:
        df["dollar_volume"] = df["close"] * df["volume"]

    # Replace ±inf introduced by ``returns`` on zero-priced rows so
    # downstream rolling stats don't blow up.
    df["returns"] = df["returns"].replace([np.inf, -np.inf], np.nan)

    return MarketPanel(frame=df, available_columns=frozenset(df.columns))


__all__ = [
    "DERIVED_COLUMNS",
    "MarketPanel",
    "OPTIONAL_INPUT_COLUMNS",
    "REQUIRED_INPUT_COLUMNS",
    "build_market_panel",
]
