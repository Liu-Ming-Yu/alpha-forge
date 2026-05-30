"""Simulated-backend parity for Arm Q (the production lead) — ADR-011 increment 5.

Reconciles the LIVE Q scoring + selection + conviction sizing against the
backtest on identical real bars, as-of the latest available date. This is the
check that caught the dollar-volume defect (increment 4: live top-30 overlapped
the backtest only 3/30 because the live raw weighted sum clamped to ties). After
the fix — the live ``pv_formulaic`` bundle rank-normalizes with the same kernel
the backtest ranker uses, and both size by the shared conviction kernel — the
expectation is **30/30 selection overlap and matching weights**.

Live path exercised (the real engine objects):
  bars → build_pv_formulaic_feature_bundle (rank-normalized) → FeatureVector(s)
  → LinearWeightSignalModel(G weights).score → LongOnlyPortfolioConstructor(
  top_n=30, conviction_shrinkage=0.25).build_targets → weights.

Backtest-equivalent (the "truth"):
  compute_pv_formulaic_frame → latest cross-section → cross_sectional_rank_normalize
  → Σ feature·G_weight → top-30 → conviction_proportions(0.25) × gross.

Run: python scripts/validate_arm_q_live_parity.py
"""

from __future__ import annotations

import glob
import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.core.algorithms.conviction import conviction_proportions
from quant_platform.core.algorithms.portfolio_construction import LongOnlyPortfolioConstructor
from quant_platform.core.domain.portfolio import RiskLimits
from quant_platform.core.domain.portfolio.positions import AccountSnapshot
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
from quant_platform.core.domain.research.features import FeatureVector
from quant_platform.core.domain.signals import RegimeLabel, RegimeState
from quant_platform.services.research_service.features.kernel.transforms import (
    cross_sectional_rank_normalize,
)
from quant_platform.services.research_service.features.pv_formulaic.compute import (
    compute_pv_formulaic_frame,
)
from quant_platform.services.research_service.features.pv_formulaic.family import (
    PV_FORMULAIC_FEATURE_SET_VERSION,
    build_pv_formulaic_feature_bundle,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = PROJECT_ROOT / "infra" / "config" / "universe_300.json"
BARS = PROJECT_ROOT / "data" / "parquet" / "bars"
G_EVIDENCE = (
    PROJECT_ROOT
    / "data/parquet/research/backtest_latest_stack_realized_v2"
    / "arm_long_only_top30_pv_formulaic_streakdial.json"
)
GROSS = Decimal("0.22")
SHRINKAGE = 0.25
TOP_N = 30


def _bars_by_instrument(years: tuple[int, ...]) -> dict[uuid.UUID, list]:
    from quant_platform.core.domain.market_data.bars import MarketBar  # noqa: PLC0415

    universe = json.loads(UNIVERSE.read_text())
    out: dict[uuid.UUID, list] = {}
    for inst in universe:
        rows: list = []
        for yr in years:
            for fp in glob.glob(str(BARS / inst / "daily" / f"{yr}.parquet")):
                df = pd.read_parquet(fp)
                df = df[df["bar_seconds"] == 86400]
                for r in df.itertuples():
                    rows.append(
                        MarketBar(
                            bar_id=uuid.uuid4(),
                            instrument_id=uuid.UUID(inst),
                            timestamp=pd.Timestamp(r.timestamp).to_pydatetime(),
                            bar_seconds=86400,
                            open=Decimal(str(r.open)),
                            high=Decimal(str(r.high)),
                            low=Decimal(str(r.low)),
                            close=Decimal(str(r.close)),
                            volume=int(r.volume),
                        )
                    )
        if rows:
            out[uuid.UUID(inst)] = sorted(rows, key=lambda b: b.timestamp)
    return out


def _ohlcv(years: tuple[int, ...]) -> pd.DataFrame:
    universe = json.loads(UNIVERSE.read_text())
    frames = []
    for inst in universe:
        for yr in years:
            for fp in glob.glob(str(BARS / inst / "daily" / f"{yr}.parquet")):
                d = pd.read_parquet(fp)
                d = d[d["bar_seconds"] == 86400].copy()
                d["instrument_id"] = inst
                d["date"] = (
                    pd.to_datetime(d["timestamp"], utc=True)
                    .dt.tz_convert("UTC")
                    .dt.normalize()
                    .dt.tz_localize(None)
                )
                frames.append(
                    d[["instrument_id", "date", "open", "high", "low", "close", "volume"]]
                )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["instrument_id", "date"])
        .drop_duplicates(["instrument_id", "date"])
        .reset_index(drop=True)
    )


def main() -> int:
    weights = json.loads(G_EVIDENCE.read_text())["selected_weights"]
    years = (2022, 2023, 2024, 2025)
    print(f"[1] Loading bars ({years}) ...")
    ohlcv = _ohlcv(years)
    as_of_date = ohlcv["date"].max()
    print(f"    {ohlcv['instrument_id'].nunique()} instruments, as-of {as_of_date.date()}")

    # ---- LIVE path (real engine objects) ----
    print("[2] LIVE: bundle → LinearWeightSignalModel → LongOnlyPortfolioConstructor ...")
    bundle = build_pv_formulaic_feature_bundle(_bars_by_instrument(years))
    as_of = datetime(as_of_date.year, as_of_date.month, as_of_date.day, tzinfo=UTC)
    run = StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="arm_q",
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=as_of,
        started_at=as_of,
    )
    vectors = [
        FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=inst,
            as_of=as_of,
            feature_set_version=PV_FORMULAIC_FEATURE_SET_VERSION,
            features=feats,
            strategy_run_id=run.run_id,
        )
        for inst, feats in bundle.alpha_features.items()
    ]
    model = LinearWeightSignalModel(weights, model_version="ic-weighted-non-negative")
    live_scores = {s.instrument_id: float(s.score) for s in model.score(vectors, run)}
    n_clamped = sum(1 for s in live_scores.values() if s in (-1.0, 1.0))
    constructor = LongOnlyPortfolioConstructor(top_n=TOP_N, conviction_shrinkage=SHRINKAGE)
    regime = RegimeState(
        regime_id=uuid.uuid4(),
        as_of=as_of,
        regime_label=RegimeLabel.RISK_ON,
        confidence=1.0,
        detector_version="parity",
        supporting_features={},
    )
    limits = RiskLimits(
        limits_id=uuid.uuid4(),
        strategy_run_id=run.run_id,
        effective_from=as_of,
        max_single_name_weight=Decimal("0.05"),
        max_sector_weight=Decimal("0.20"),
        max_gross_exposure=GROSS,
        max_daily_turnover=Decimal("0.20"),
        min_cash_buffer=Decimal("0.05"),
        max_drawdown_halt=Decimal("-0.15"),
        vol_target_annualised=None,
    )
    account = AccountSnapshot(
        snapshot_id=uuid.uuid4(),
        as_of=as_of,
        settled_cash=Decimal("50000"),
        unsettled_cash=Decimal("0"),
        reserved_cash=Decimal("0"),
        available_cash=Decimal("50000"),
        net_asset_value=Decimal("50000"),
        positions=(),
    )
    scores_for_ctor = sorted(model.score(vectors, run), key=lambda s: s.score, reverse=True)
    live_target = constructor.build_targets(scores_for_ctor, regime, account, limits)
    live_weights = {k: float(v) for k, v in live_target.weights.items()}

    # ---- BACKTEST-equivalent (the "truth") ----
    print("[3] BACKTEST-equiv: rank-norm → Σ·G_weight → top-30 → conviction ...")
    frame = compute_pv_formulaic_frame(ohlcv)
    feats = list(weights)
    latest = frame.sort_values("date").groupby("instrument_id", sort=False).tail(1)
    normed = cross_sectional_rank_normalize(latest, feats, date_column="date").set_index(
        "instrument_id"
    )
    bt_score = normed[feats].mul(pd.Series(weights)).sum(axis=1)
    bt_score.index = [uuid.UUID(i) for i in bt_score.index]
    bt_top = list(bt_score.sort_values(ascending=False).head(TOP_N).index)
    bt_props = conviction_proportions(
        [float(bt_score[i]) for i in bt_top], shrinkage=SHRINKAGE, reference="min"
    )
    bt_weights = {i: p * float(GROSS) for i, p in zip(bt_top, bt_props, strict=True)}

    # ---- Reconcile ----
    live_top = set(live_weights)
    bt_top_set = set(bt_top)
    overlap = len(live_top & bt_top_set)
    # score parity on the shared names
    score_pairs = [(live_scores[i], float(bt_score[i])) for i in bt_top if i in live_scores]
    score_corr = (
        float(np.corrcoef([a for a, _ in score_pairs], [b for _, b in score_pairs])[0, 1])
        if len(score_pairs) > 1
        else float("nan")
    )
    weight_l1 = sum(
        abs(live_weights.get(i, 0.0) - bt_weights.get(i, 0.0)) for i in bt_top_set | live_top
    )

    print("\n================= ARM Q LIVE-vs-BACKTEST PARITY =================")
    print(
        f"live scores clamped to ±1 (dollar-volume tie symptom): {n_clamped} / {len(live_scores)}"
    )
    print(f"top-{TOP_N} selection overlap:        {overlap}/{TOP_N}")
    print(f"score correlation (shared names):   {score_corr:.6f}")
    print(f"weight L1 distance (live vs bt):     {weight_l1:.6f}")
    print(
        f"live gross:  {sum(live_weights.values()):.4f}   "
        f"backtest gross: {sum(bt_weights.values()):.4f}"
    )
    ok = overlap == TOP_N and weight_l1 < 1e-3 and n_clamped == 0
    print(f"\nPARITY: {'PASS' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
