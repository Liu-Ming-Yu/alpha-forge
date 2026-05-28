"""Research-side IO helpers.

These are convenience loaders for ad-hoc research scripts that need to read
artifacts produced by the data and feature services without going through the
async session-bound stores. Anything that talks to live infrastructure or
joins the supervised-learning pipeline should still go through
``services.data_service`` / ``services.research_service`` -- this package only
exists so a one-off walk-forward or notebook does not reinvent (and quietly
get wrong) basic filtering invariants like "daily bars only" on the parquet
bar store.

See :mod:`quant_platform.research.io.bars` for the bar reader.
"""

from __future__ import annotations

from quant_platform.research.io.bars import load_bars, load_daily_bars

__all__ = ["load_bars", "load_daily_bars"]
