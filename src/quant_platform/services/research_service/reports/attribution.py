"""Portfolio P&L attribution decomposition.

Splits the realised per-rebalance P&L of a backtest into per-factor,
per-sector, and per-regime contributions.  The result is an
``AttributionArtifact`` persisted as JSON under
``data/backtest/<run_id>/attribution.json`` and consumed by the
tearsheet renderer (Phase 2.6).

Decomposition conventions:
    - Per-factor contribution: for each rebalance we project the
      portfolio target weights onto the factor exposures used in
      ``LinearWeightSignalModel`` and multiply by the realised asset
      return over the cycle.  Sum over rebalances gives the factor's
      gross contribution to realised P&L.
    - Per-sector contribution: sum (weight x return) across instruments
      grouped by ``sector_map[instrument_id]``; unmapped instruments
      fall into ``"UNMAPPED"`` so the preflight from the Parity sprint
      still surfaces their contribution.
    - Per-regime contribution: group rebalances by
      ``regime_by_rebalance[as_of]`` and sum the per-rebalance portfolio
      return; a non-covered rebalance falls into ``"unknown"``.

The inputs are intentionally plain mappings rather than domain objects
so offline research notebooks can produce an artifact from Parquet
replays without requiring a full session wiring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from pathlib import Path

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AttributionCycle:
    """Inputs for a single rebalance's attribution.

    ``target_weights`` and ``realized_returns`` share the same set of
    instrument ids.  ``factor_exposures`` maps instrument_id ->
    factor_name -> rank-normalised score (same domain used for signals).
    Absent factor names are treated as zero.
    """

    as_of: datetime
    target_weights: Mapping[uuid.UUID, float]
    realized_returns: Mapping[uuid.UUID, float]
    factor_exposures: Mapping[uuid.UUID, Mapping[str, float]]
    regime_label: str


@dataclass(frozen=True)
class AttributionArtifact:
    """All three decompositions + the cycle-level detail for audit."""

    run_id: uuid.UUID
    as_of: datetime
    factor_pnl: Mapping[str, float]
    sector_pnl: Mapping[str, float]
    regime_pnl: Mapping[str, float]
    total_pnl: float
    num_cycles: int
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["run_id"] = str(self.run_id)
        payload["as_of"] = self.as_of.astimezone(UTC).isoformat()
        return json.dumps(payload, indent=2, sort_keys=True)


def compute_attribution(
    run_id: uuid.UUID,
    cycles: Sequence[AttributionCycle],
    sector_map: Mapping[uuid.UUID, str],
    *,
    as_of: datetime | None = None,
    factor_weights: Mapping[str, float] | None = None,
) -> AttributionArtifact:
    """Build the per-factor / per-sector / per-regime decomposition.

    ``factor_weights`` (optional) is used to scale each factor's raw
    exposure x return contribution so the sum across factors matches
    the actual portfolio return more closely.  When ``None`` the raw
    exposure x return product is reported, which is the right choice
    for a signal-attribution view where factor weights are the object
    of investigation.
    """
    factor_totals: dict[str, float] = {}
    sector_totals: dict[str, float] = {}
    regime_totals: dict[str, float] = {}
    total_pnl = 0.0
    sum_weight_count = 0

    for cycle in cycles:
        cycle_pnl = 0.0
        for iid, w in cycle.target_weights.items():
            ret = cycle.realized_returns.get(iid)
            if ret is None:
                continue
            contrib = float(w) * float(ret)
            cycle_pnl += contrib
            sector = sector_map.get(iid, "UNMAPPED")
            sector_totals[sector] = sector_totals.get(sector, 0.0) + contrib
            exposures = cycle.factor_exposures.get(iid, {})
            for factor_name, exposure in exposures.items():
                scale = (
                    factor_weights[factor_name]
                    if factor_weights and factor_name in factor_weights
                    else 1.0
                )
                factor_contrib = float(exposure) * float(ret) * float(w) * scale
                factor_totals[factor_name] = factor_totals.get(factor_name, 0.0) + factor_contrib
            sum_weight_count += 1
        regime_totals[cycle.regime_label] = regime_totals.get(cycle.regime_label, 0.0) + cycle_pnl
        total_pnl += cycle_pnl

    when = as_of or (cycles[-1].as_of if cycles else datetime.now(tz=UTC))
    return AttributionArtifact(
        run_id=run_id,
        as_of=when,
        factor_pnl=factor_totals,
        sector_pnl=sector_totals,
        regime_pnl=regime_totals,
        total_pnl=total_pnl,
        num_cycles=len(cycles),
        metadata={
            "sum_weight_count": sum_weight_count,
            "used_factor_weights": factor_weights is not None,
        },
    )


def write_attribution(
    artifact: AttributionArtifact,
    root: Path,
) -> Path:
    """Write ``attribution.json`` under ``root/<run_id>/``."""
    directory = root / str(artifact.run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "attribution.json"
    path.write_text(artifact.to_json(), encoding="utf-8")
    log.info(
        "attribution.artifact_written",
        path=str(path),
        factors=len(artifact.factor_pnl),
        sectors=len(artifact.sector_pnl),
        regimes=len(artifact.regime_pnl),
        total_pnl=artifact.total_pnl,
    )
    return path


def read_attribution(path: Path) -> dict[str, object]:
    """Load the attribution artifact as a plain dict."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}
