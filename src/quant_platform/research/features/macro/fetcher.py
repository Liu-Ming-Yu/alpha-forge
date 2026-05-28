"""Operator-only helper: fetch FRED series and return
:class:`MacroSeriesValue` records ready for ``compute_macro_features``.

This module is **NOT** part of the family contract — the family
itself takes :class:`MacroSeriesValue` iterables and is feed-agnostic.
The fetcher is here as a convenience for the common case: an operator
who has a FRED API key (free at https://fred.stlouisfed.org/) and
wants to populate the family without writing their own adapter.

Lazy-imports ``fredapi``: a) so the family stays light-dependency,
b) so the test suite doesn't need ``fredapi`` installed. The import
happens inside :func:`fetch_fred_series`, so the test surface never
touches it.

Operator usage:

    from quant_platform.research.features.macro.fetcher import fetch_fred_series
    from quant_platform.research.features.macro.config import REQUIRED_SERIES_IDS

    records = fetch_fred_series(
        series_ids=REQUIRED_SERIES_IDS,
        start_date="2020-01-01",
        end_date="2025-01-01",
        api_key="<FRED_API_KEY>",  # or read from env QP__MACRO__FRED_API_KEY
    )
    # records is a list[MacroSeriesValue] ready for compute_macro_features

The fetcher is documented in `docs/macro-v1-family.md`'s "Operator
quickstart" section.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.research.features.macro.schemas import MacroSeriesValue

if TYPE_CHECKING:
    from collections.abc import Sequence


def fetch_fred_series(
    *,
    series_ids: Sequence[str],
    start_date: str,
    end_date: str,
    api_key: str | None = None,
) -> list[MacroSeriesValue]:
    """Fetch FRED series and return them as MacroSeriesValue records.

    Parameters
    ----------
    series_ids:
        Sequence of FRED series IDs to fetch. Pass
        :data:`~.config.REQUIRED_SERIES_IDS` to populate the v1 family.
    start_date, end_date:
        ISO date strings (``YYYY-MM-DD``). Inclusive.
    api_key:
        FRED API key (free, sign up at
        https://fred.stlouisfed.org/docs/api/api_key.html). If
        omitted, reads the ``FRED_API_KEY`` environment variable.

    Returns
    -------
    list[MacroSeriesValue]
        One record per observation per series. Dates are converted
        to UTC midnight timestamps. NaN observations are filtered
        out (the schema rejects NaN values; this is the boundary).

    Raises
    ------
    ImportError
        When ``fredapi`` is not installed. The error message points
        the operator at ``pip install fredapi``.
    """
    try:
        from fredapi import Fred  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "fetch_fred_series requires the 'fredapi' package. Install it "
            "with `pip install fredapi`. The macro-v1 family itself does "
            "NOT depend on fredapi — only this convenience fetcher."
        ) from exc

    fred = Fred(api_key=api_key)
    records: list[MacroSeriesValue] = []
    for sid in series_ids:
        series = fred.get_series(sid, observation_start=start_date, observation_end=end_date)
        for observation_date, value in series.items():
            # Skip NaN observations — FRED returns them for holidays
            # and pre-publication dates. The schema rejects them at
            # the boundary anyway.
            if value != value:  # NaN check
                continue
            # FRED returns naive datetime; tag as UTC at midnight.
            if observation_date.tzinfo is None:  # type: ignore[union-attr]
                ts = observation_date.replace(tzinfo=UTC)  # type: ignore[union-attr]
            else:
                ts = observation_date
            records.append(
                MacroSeriesValue(
                    series_id=sid,
                    observation_date=ts if isinstance(ts, datetime) else ts.to_pydatetime(),
                    value=float(value),
                )
            )
    return records


__all__ = ["fetch_fred_series"]
