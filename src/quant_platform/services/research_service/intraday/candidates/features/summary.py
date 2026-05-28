"""Coverage diagnostics for intraday candidate screens."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from quant_platform.core.domain.market_data import INTRADAY_BAR_SECONDS
from quant_platform.services.research_service.campaigns.screening.common import (
    REQUIRED_INSTRUMENTS,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def intraday_source_summary(
    *,
    intraday_bars: Sequence[MarketBar],
    samples: Sequence[SupervisedAlphaSample],
) -> dict[str, object]:
    sample_instruments = {sample.instrument_id for sample in samples}
    observed_instruments = {bar.instrument_id for bar in intraday_bars}
    covered = observed_instruments & sample_instruments
    non_1m = sum(1 for bar in intraday_bars if bar.bar_seconds != INTRADAY_BAR_SECONDS)
    required = min(len(sample_instruments), REQUIRED_INSTRUMENTS)
    rows_by_instrument: dict[str, int] = defaultdict(int)
    for bar in intraday_bars:
        rows_by_instrument[str(bar.instrument_id)] += 1
    blockers: list[str] = []
    if not intraday_bars:
        blockers.append("intraday bar coverage 0")
    if non_1m:
        blockers.append(f"intraday bars not 1-minute: {non_1m}")
    if len(covered) < required:
        blockers.append(f"intraday sample-instrument coverage {len(covered)} < {required}")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "bar_count": len(intraday_bars),
        "instrument_count": len(observed_instruments),
        "sample_instrument_count": len(sample_instruments),
        "covered_sample_instruments": len(covered),
        "required_sample_instruments": required,
        "rows_by_instrument": dict(sorted(rows_by_instrument.items())),
    }
