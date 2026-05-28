"""Walk-forward out-of-sample evaluation of the XGBoost ranker.

The research campaign trains XGBoost on a single train/validation split and
records only a validation IC — it never runs XGBoost through the walk-forward
eligibility path (OOS IC, drawdown, IC streak, Sharpe). This script fills that
gap: per-fold GPU training with a purge gap, OOS prediction, and the *same*
long-only portfolio evaluation the campaign uses for the linear model — so the
XGBoost numbers are directly comparable to a campaign ``eligibility.json``.

Usage::

    python scripts/xgboost_walkforward.py \
        --samples data/parquet/research/_inputs/u337_samples.json
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from collections import defaultdict
from datetime import datetime

import numpy as np
import xgboost as xgb

from quant_platform.services.research_service.campaigns.portfolio.evaluation import (
    evaluate_long_only_portfolio,
)
from quant_platform.services.research_service.campaigns.portfolio.types import (
    CampaignPortfolioConfig,
)
from quant_platform.services.research_service.reports.statistics import negative_streak
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

# Per-fold XGBoost hyperparameters (regularized; ~best_iteration from the
# campaign's conservative GPU search).
_PARAMS = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg",
    "ndcg_exp_gain": False,
    "tree_method": "hist",
    "device": "cuda",
    "eta": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "seed": 17,
}
_NUM_ROUNDS = 50


def load_samples(path: str) -> list[SupervisedAlphaSample]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    rows = raw if isinstance(raw, list) else raw.get("samples", [])
    out: list[SupervisedAlphaSample] = []
    for r in rows:
        out.append(
            SupervisedAlphaSample(
                as_of=datetime.fromisoformat(str(r["as_of"])),
                instrument_id=uuid.UUID(str(r["instrument_id"])),
                features={k: float(v) for k, v in r["features"].items()},
                forward_return=float(r["forward_return"]),
            )
        )
    return out


def _dmatrix(samples: list[SupervisedAlphaSample], feature_names: list[str]) -> xgb.DMatrix:
    """Build a grouped DMatrix with per-date rank-index relevance labels."""
    by_day: dict[datetime, list[SupervisedAlphaSample]] = defaultdict(list)
    for s in samples:
        by_day[s.as_of].append(s)
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    for day in sorted(by_day):
        group = by_day[day]
        ranked = sorted({s.forward_return for s in group})
        rank_of = {v: i for i, v in enumerate(ranked)}
        groups.append(len(group))
        for s in group:
            rows.append([s.features.get(n, 0.0) for n in feature_names])
            labels.append(rank_of[s.forward_return])
    dmat = xgb.DMatrix(np.asarray(rows, dtype=float), label=np.asarray(labels, dtype=float))
    dmat.set_group(np.asarray(groups, dtype=np.uint32))
    return dmat


def walk_forward(
    samples: list[SupervisedAlphaSample],
    *,
    train_days: int,
    test_days: int,
    purge_days: int,
) -> list[tuple[SupervisedAlphaSample, float]]:
    """Train XGBoost per fold and return OOS (sample, prediction) pairs."""
    by_day: dict[datetime, list[SupervisedAlphaSample]] = defaultdict(list)
    for s in samples:
        by_day[s.as_of].append(s)
    dates = sorted(by_day)
    feature_names = sorted({n for s in samples for n in s.features})
    scored: list[tuple[SupervisedAlphaSample, float]] = []

    start = train_days + purge_days
    fold = 0
    for t in range(start, len(dates), test_days):
        train_dates = dates[max(0, t - purge_days - train_days) : t - purge_days]
        test_dates = dates[t : t + test_days]
        if not train_dates or not test_dates:
            continue
        train = [s for d in train_dates for s in by_day[d]]
        test = [s for d in test_dates for s in by_day[d]]
        booster = xgb.train(_PARAMS, _dmatrix(train, feature_names), num_boost_round=_NUM_ROUNDS)
        test_x = xgb.DMatrix(
            np.asarray([[s.features.get(n, 0.0) for n in feature_names] for s in test], dtype=float)
        )
        preds = booster.predict(test_x)
        scored.extend(zip(test, (float(p) for p in preds), strict=True))
        fold += 1
    print(f"walk-forward: {fold} folds, {len(scored)} OOS predictions")
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward XGBoost evaluation.")
    parser.add_argument("--samples", default="data/parquet/research/_inputs/u337_samples.json")
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--purge-days", type=int, default=21)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--max-gross", type=float, default=0.45)
    args = parser.parse_args()

    samples = load_samples(args.samples)
    print(f"loaded {len(samples)} samples")
    scored = walk_forward(
        samples,
        train_days=args.train_days,
        test_days=args.test_days,
        purge_days=args.purge_days,
    )

    config = CampaignPortfolioConfig(
        top_n=args.top_n,
        max_gross_exposure=args.max_gross,
        max_single_name_weight=0.05,
    )
    result = evaluate_long_only_portfolio(scored, slippage_bps_per_turnover=10.0, config=config)

    ics = [ic for _, ic in result.daily_ics]
    returns = list(result.daily_returns)
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        max_dd = min(max_dd, v / peak - 1.0)
    mean_r = sum(returns) / len(returns) if returns else 0.0
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns)) if returns else 0.0
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    mean_ic = sum(ics) / len(ics) if ics else 0.0

    print("\n=== Walk-forward XGBoost OOS result ===")
    print(f"  oos_mean_ic            {mean_ic:+.4f}")
    print(f"  ic_negative_streak     {negative_streak(ics):.0f}")
    print(f"  annualised_sharpe      {sharpe:+.3f}")
    print(f"  max_drawdown           {max_dd:+.3f}")
    print(f"  total_return           {equity[-1] - 1.0:+.3f}")
    print(f"  observations           {len(returns)}")


if __name__ == "__main__":
    main()
