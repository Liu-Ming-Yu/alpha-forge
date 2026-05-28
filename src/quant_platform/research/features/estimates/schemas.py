"""Input record schemas for the ``estimates-v1`` feature family.

The platform doesn't yet have a wired analyst-estimate feed (IBES,
FactSet, Visible Alpha — all paid vendor products). The family is
scaffolded around explicit dataclass contracts that the operator can
populate from any vendor in a follow-up PR. Tests use synthetic
fixtures.

Two input streams:

* :class:`ConsensusSnapshot` — daily consensus snapshot for a
  ``(instrument, target_period, estimate_kind)`` triple, carrying
  mean / std / count plus a 30-day rolling count of revisions.
  Matches the shape of the IBES Summary file (one row per
  instrument-period per day) or the Sharadar / S&P Capital IQ
  equivalents.
* :class:`EarningsSurpriseRecord` — one historical earnings-actual
  vs consensus event, with the consensus the actuals are compared
  against and the date the actuals were publicly reported. PIT-safe:
  the family only consumes records where ``reported_at <= panel
  date``.

Stability contract: every schema is frozen + range-validated in
``__post_init__``. Adding / renaming a field is a v2 bump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


#: Valid ``estimate_kind`` values. ``eps`` is per-share earnings
#: estimate; ``revenue`` is sales estimate. Other kinds (EBITDA, FCF)
#: would land in v2 once we know they're worth carrying.
ALLOWED_ESTIMATE_KINDS: tuple[str, ...] = ("eps", "revenue")

#: Valid ``target_period`` values. ``FY1`` = next fiscal year's
#: estimate; ``FY2`` = year after that. ``Q1``/``Q2``/``Q3``/``Q4`` are
#: the next four quarterly estimates. The catalogue is small on
#: purpose — IBES uses a longer set of periodicities, but v1 focuses
#: on the ones with the most analyst coverage.
ALLOWED_TARGET_PERIODS: tuple[str, ...] = (
    "FY1",
    "FY2",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
)


@dataclass(frozen=True)
class ConsensusSnapshot:
    """One daily analyst-consensus snapshot.

    Attributes
    ----------
    instrument_id:
        Instrument the consensus is for.
    snapshot_date:
        The date the consensus is as-of. Daily cadence is the IBES
        Summary file convention. Tz-aware (UTC recommended).
    target_period:
        Which fiscal period the estimate covers. See
        :data:`ALLOWED_TARGET_PERIODS`.
    estimate_kind:
        ``"eps"`` or ``"revenue"``. See :data:`ALLOWED_ESTIMATE_KINDS`.
    mean_estimate:
        Cross-analyst mean of currently-active estimates.
    std_estimate:
        Cross-analyst standard deviation. ``None`` is allowed when
        only one analyst is covering (std is undefined for n=1) —
        the dispersion feature returns NaN on that branch.
    n_estimates:
        Number of analysts contributing to this snapshot. Strictly
        positive (zero would have no consensus to record).
    n_up_revisions_30d:
        Count of analysts who revised UP in the trailing 30 days.
        Non-negative.
    n_down_revisions_30d:
        Count of analysts who revised DOWN in the trailing 30 days.
        Non-negative.
    """

    instrument_id: str
    snapshot_date: datetime
    target_period: str
    estimate_kind: str
    mean_estimate: float
    std_estimate: float | None
    n_estimates: int
    n_up_revisions_30d: int = 0
    n_down_revisions_30d: int = 0

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("ConsensusSnapshot.instrument_id must be non-empty")
        if self.snapshot_date.tzinfo is None:
            raise ValueError("ConsensusSnapshot.snapshot_date must be timezone-aware")
        if self.target_period not in ALLOWED_TARGET_PERIODS:
            raise ValueError(
                f"ConsensusSnapshot.target_period must be one of "
                f"{ALLOWED_TARGET_PERIODS!r}; got {self.target_period!r}"
            )
        if self.estimate_kind not in ALLOWED_ESTIMATE_KINDS:
            raise ValueError(
                f"ConsensusSnapshot.estimate_kind must be one of "
                f"{ALLOWED_ESTIMATE_KINDS!r}; got {self.estimate_kind!r}"
            )
        if self.n_estimates <= 0:
            raise ValueError(f"ConsensusSnapshot.n_estimates must be > 0; got {self.n_estimates}")
        if self.std_estimate is not None and self.std_estimate < 0:
            raise ValueError(
                f"ConsensusSnapshot.std_estimate must be >= 0 or None; got {self.std_estimate}"
            )
        if self.n_up_revisions_30d < 0:
            raise ValueError(
                f"ConsensusSnapshot.n_up_revisions_30d must be >= 0; got {self.n_up_revisions_30d}"
            )
        if self.n_down_revisions_30d < 0:
            raise ValueError(
                f"ConsensusSnapshot.n_down_revisions_30d must be >= 0; "
                f"got {self.n_down_revisions_30d}"
            )


@dataclass(frozen=True)
class EarningsSurpriseRecord:
    """One historical earnings-actual vs consensus snapshot.

    Attributes
    ----------
    instrument_id:
        Instrument.
    fiscal_period_end:
        Period the actuals are for (typically quarter-end).
    actual_eps:
        Reported EPS.
    consensus_mean_eps:
        Consensus mean EPS at the time the actuals were reported —
        i.e. the consensus the actuals are being compared against,
        NOT the current latest consensus. v1 uses % surprise
        (``(actual - consensus_mean) / |consensus_mean|``), so a
        zero consensus produces NaN.
    consensus_std_eps:
        Consensus std at the time of reporting. ``None`` allowed for
        single-analyst coverage. Currently unused by v1 features but
        carried so future versions can compute z-score surprises.
    reported_at:
        When the actuals were publicly reported. PIT-safe: the
        family only consumes records where ``reported_at`` is on or
        before the panel date.
    """

    instrument_id: str
    fiscal_period_end: datetime
    actual_eps: float
    consensus_mean_eps: float
    consensus_std_eps: float | None
    reported_at: datetime

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("EarningsSurpriseRecord.instrument_id must be non-empty")
        if self.fiscal_period_end.tzinfo is None:
            raise ValueError("EarningsSurpriseRecord.fiscal_period_end must be timezone-aware")
        if self.reported_at.tzinfo is None:
            raise ValueError("EarningsSurpriseRecord.reported_at must be timezone-aware")
        if self.reported_at < self.fiscal_period_end:
            raise ValueError("EarningsSurpriseRecord.reported_at must be >= fiscal_period_end")
        if self.consensus_std_eps is not None and self.consensus_std_eps < 0:
            raise ValueError(
                f"EarningsSurpriseRecord.consensus_std_eps must be >= 0 or None; "
                f"got {self.consensus_std_eps}"
            )


__all__ = [
    "ALLOWED_ESTIMATE_KINDS",
    "ALLOWED_TARGET_PERIODS",
    "ConsensusSnapshot",
    "EarningsSurpriseRecord",
]
