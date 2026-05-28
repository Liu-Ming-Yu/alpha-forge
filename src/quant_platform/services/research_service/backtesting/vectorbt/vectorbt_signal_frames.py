"""Signal-frame construction for the VectorBT research backtest."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd

from quant_platform.core.domain.research import FeatureVector, StrategyRun

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.contracts import PortfolioConstructor, SignalModel
    from quant_platform.core.domain.signals import RegimeState


def build_vectorbt_signal_frames(
    *,
    rebalance_timestamps: list[datetime],
    feature_series: dict[datetime, dict[uuid.UUID, dict[str, float]]],
    price_series: dict[datetime, dict[uuid.UUID, Decimal]],
    strategy_run: StrategyRun,
    regime_series: dict[datetime, RegimeState],
    signal_model: SignalModel | None,
    portfolio_constructor: PortfolioConstructor,
    parity_mode: bool,
    top_n: int,
) -> dict[uuid.UUID, pd.DataFrame]:
    """Build per-instrument close/signal/regime-scale frames."""
    all_instr_ids: set[uuid.UUID] = set()
    for ts in rebalance_timestamps:
        all_instr_ids.update(feature_series.get(ts, {}).keys())
        all_instr_ids.update(price_series.get(ts, {}).keys())

    last_price: dict[uuid.UUID, float] = {}
    signals_by_instr: dict[uuid.UUID, list[float]] = {i: [] for i in all_instr_ids}
    regime_scales_by_instr: dict[uuid.UUID, list[float]] = {i: [] for i in all_instr_ids}
    close_by_instr: dict[uuid.UUID, list[float]] = {i: [] for i in all_instr_ids}

    for ts in rebalance_timestamps:
        prices_at_ts = price_series.get(ts, {})
        features_at_ts = feature_series.get(ts, {})
        regime_state = regime_series.get(ts)

        scale = Decimal("1.0") if parity_mode else Decimal("0.75")
        if (
            not parity_mode
            and regime_state is not None
            and hasattr(portfolio_constructor, "scale_for_regime")
        ):
            scale = portfolio_constructor.scale_for_regime(regime_state.regime_label)

        for instr_id in all_instr_ids:
            price = prices_at_ts.get(instr_id)
            if price is not None:
                last_price[instr_id] = float(price)
            close_by_instr[instr_id].append(last_price.get(instr_id, 0.0))

        raw_scores: dict[uuid.UUID, float] = {}
        for instr_id in all_instr_ids:
            feat_dict = features_at_ts.get(instr_id, {})
            if feat_dict and signal_model is not None:
                fv = FeatureVector(
                    vector_id=uuid.uuid4(),
                    instrument_id=instr_id,
                    as_of=ts,
                    feature_set_version="0.0.0",
                    features=feat_dict,
                    strategy_run_id=strategy_run.run_id,
                )
                scored = signal_model.score([fv], strategy_run)
                if scored:
                    raw_scores[instr_id] = scored[0].score

        positive_ranked = sorted(
            [(iid, score) for iid, score in raw_scores.items() if score > 0],
            key=lambda item: item[1],
            reverse=True,
        )
        top_n_ids = {iid for iid, _ in positive_ranked[:top_n]}

        for instr_id in all_instr_ids:
            signals_by_instr[instr_id].append(1.0 if instr_id in top_n_ids else 0.0)
            regime_scales_by_instr[instr_id].append(float(scale))

    frames: dict[uuid.UUID, pd.DataFrame] = {}
    index = pd.DatetimeIndex(rebalance_timestamps, tz="UTC")
    for instr_id in all_instr_ids:
        frames[instr_id] = pd.DataFrame(
            {
                "close": close_by_instr[instr_id],
                "signal": signals_by_instr[instr_id],
                "regime_scale": regime_scales_by_instr[instr_id],
            },
            index=index,
        )
    return frames
