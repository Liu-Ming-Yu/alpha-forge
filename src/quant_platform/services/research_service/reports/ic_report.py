"""Information Coefficient and alpha-decay reporting.

Given per-rebalance (feature, forward-return) panels, this module
produces a rolling Spearman IC series for each factor and a per-horizon
decay curve (1 / 5 / 10 / 20 day forward returns).  The output feeds
the tearsheet and the operator API ``/research/ic/{run_id}`` endpoint.

IC is measured per-rebalance using cross-sectional Spearman rank
correlation between the factor score and the forward return.  Rolling
mean + standard deviation over configurable windows give the reader a
sense of stability; the per-horizon mean IC is the alpha-decay curve.

Design notes:
    - No scipy dependency.  Spearman correlation is computed via
      ``numpy.argsort`` ranks and ``numpy.corrcoef``.
    - Numeric instability (all-equal ranks, fewer than two non-NaN
      observations) is handled explicitly: the per-rebalance IC is
      recorded as NaN, which the rolling statistics carry through
      unchanged.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import structlog

from quant_platform.services.research_service.reports.ic_report_models import (
    ICPanel,
    ICReport,
    ICSeries,
)
from quant_platform.services.research_service.reports.statistics import (
    average_ranks as _shared_average_ranks,
)
from quant_platform.services.research_service.reports.statistics import (
    spearman_ic,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from pathlib import Path

log = structlog.get_logger(__name__)


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Cross-sectional Spearman rank correlation.

    Returns NaN when the inputs have fewer than 2 non-NaN overlapping
    observations or all ranks are tied.  Implementation uses average
    ranks so ties do not distort the correlation.
    """
    return spearman_ic(
        a,
        b,
        invalid_value=float("nan"),
        constant_value=float("nan"),
        drop_non_finite=True,
    )


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Compatibility wrapper returning NumPy ranks with shared tie handling."""
    return np.asarray(_shared_average_ranks([float(value) for value in x]), dtype=float)


def _rolling(series: Sequence[float], window: int, stat: str) -> list[float]:
    """Simple rolling statistic ignoring NaN; returns NaN until window filled."""
    out: list[float] = []
    buf: list[float] = []
    for val in series:
        buf.append(val)
        if len(buf) < window:
            out.append(float("nan"))
            continue
        slice_ = np.asarray(buf[-window:], dtype=float)
        finite = slice_[np.isfinite(slice_)]
        if len(finite) == 0:
            out.append(float("nan"))
        elif stat == "mean":
            out.append(float(finite.mean()))
        elif stat == "std":
            out.append(float(finite.std(ddof=0)))
        else:
            raise ValueError(f"unknown stat: {stat}")
    return out


def compute_ic_report(
    run_id: uuid.UUID,
    panels_by_horizon: Mapping[int, Sequence[ICPanel]],
    *,
    factors: Sequence[str],
    as_of: datetime | None = None,
    rolling_window: int = 20,
) -> ICReport:
    """Build an ``ICReport`` from pre-panelled inputs.

    ``panels_by_horizon`` maps forward-return horizon (in days) to the
    matching panel series (same feature snapshot, different forward
    return windows).  The primary IC time series is taken from the
    shortest horizon; the other horizons are collapsed into the
    ``decay`` map as mean IC per horizon.
    """
    if not panels_by_horizon:
        raise ValueError("panels_by_horizon must not be empty")

    horizons = tuple(sorted(panels_by_horizon.keys()))
    primary_horizon = horizons[0]
    primary_panels = panels_by_horizon[primary_horizon]

    series_out: list[ICSeries] = []
    for factor in factors:
        timestamps: list[datetime] = []
        ic_values: list[float] = []
        for panel in primary_panels:
            factor_scores: list[float] = []
            returns: list[float] = []
            for iid, fwd in panel.forward_returns.items():
                feats = panel.features.get(iid)
                if feats is None or factor not in feats:
                    continue
                factor_scores.append(float(feats[factor]))
                returns.append(float(fwd))
            timestamps.append(panel.as_of)
            ic_values.append(_spearman(factor_scores, returns))

        decay: dict[int, float] = {}
        for horizon, panels in panels_by_horizon.items():
            per_panel_ic: list[float] = []
            for panel in panels:
                scores: list[float] = []
                rets: list[float] = []
                for iid, fwd in panel.forward_returns.items():
                    feats = panel.features.get(iid)
                    if feats is None or factor not in feats:
                        continue
                    scores.append(float(feats[factor]))
                    rets.append(float(fwd))
                per_panel_ic.append(_spearman(scores, rets))
            arr = np.asarray(per_panel_ic, dtype=float)
            arr = arr[np.isfinite(arr)]
            decay[horizon] = float(arr.mean()) if len(arr) else float("nan")

        series_out.append(
            ICSeries(
                factor=factor,
                timestamps=tuple(timestamps),
                ic=tuple(ic_values),
                rolling_mean_20=tuple(_rolling(ic_values, rolling_window, "mean")),
                rolling_std_20=tuple(_rolling(ic_values, rolling_window, "std")),
                decay=decay,
            )
        )

    when = as_of or datetime.now(tz=UTC)
    return ICReport(
        run_id=run_id,
        as_of=when,
        horizons=horizons,
        series=tuple(series_out),
        metadata={
            "rolling_window": rolling_window,
            "factor_count": len(factors),
            "primary_horizon_days": primary_horizon,
        },
    )


def write_ic_report(report: ICReport, root: Path) -> Path:
    """Write ``ic_report.json`` under ``root/<run_id>/`` and return the path."""
    directory = root / str(report.run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ic_report.json"
    path.write_text(report.to_json(), encoding="utf-8")
    log.info(
        "ic_report.artifact_written",
        path=str(path),
        factors=len(report.series),
        horizons=list(report.horizons),
    )
    return path


def read_ic_report(path: Path) -> dict[str, object]:
    """Load an IC report JSON artifact.

    Returned as a plain dict rather than an ``ICReport`` because the
    operator API endpoint only needs structured JSON; callers that want
    the typed object should reconstruct it from the dict.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}
